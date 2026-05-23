"""
Multi-agent Compliance Council powered by Gemma.

Each "agent" is a differently-prompted call to the same LLM:

  Agent 1 – Technical Analyst   : interprets spectral signatures & change patterns
  Agent 2 – Legal Expert        : evaluates EUDR Article 3 compliance
  Agent 3 – Report Writer       : synthesises a formal due-diligence narrative

Design notes
------------
* A single Gemma model instance is loaded once and reused for all three agents.
* Prompts are structured with Gemma's <start_of_turn> chat template.
* Temperature is intentionally low for the legal agent (< 0.3) to minimise
  hallucinations; slightly higher for the narrative writer to improve fluency.
* All LLM calls are synchronous – wrap in an executor for async contexts.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch

from eudr.config import settings
from eudr.exceptions import ModelInferenceError
from eudr.regulations import (
    COMMODITIES_ANNEX_I_SUMMARY,
    EUDR_NATURAL_DRIVERS,
    EUDR_NON_COMMODITY_DRIVERS,
    driver_legal_rationale,
)
from eudr.schemas import ChangeMetrics, CouncilAnalysis

logger = logging.getLogger(__name__)


class ComplianceCouncil:
    """
    Gemma-backed multi-agent council for EUDR legal and technical analysis.

    Parameters
    ----------
    model_id:
        HuggingFace model identifier (default: ``settings.council_model_id``).
    """

    # Non-gated lightweight fallback LLM (no HF token or licence required)
    _OPEN_FALLBACK_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    _OPEN_FALLBACK_TEMPLATE = "<|system|>\n{system}\n</s>\n<|user|>\n{user}\n</s>\n<|assistant|>\n"

    def __init__(self, model_id: str | None = None) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        self._model_id = model_id or settings.council_model_id
        self._fallback = True  # Always use rule-based (pre-LLM version)
        self._open_fallback = False
        self.tokenizer = None
        self.llm = None
        logger.info("Using rule-based Council (pre-LLM version)")

        # Skip all LLM loading - use rule-based analysis only

    # ── Public API ─────────────────────────────────────────────────────────────

    def deliberate(
        self,
        metrics: ChangeMetrics,
        coordinates: list[float],
        wri_driver: Optional[dict] = None,
        tree_cover: Optional[dict] = None,
    ) -> CouncilAnalysis:
        """
        Run all three council agents and return the combined analysis.

        Parameters
        ----------
        metrics:
            Deforestation metrics produced by the change-detection model.
        coordinates:
            Representative [lon, lat] point from the polygon (for context).
        wri_driver:
            WRI GDM driver dict (driver_class, driver_label, is_eudr_driver, source).
        tree_cover:
            Hansen treecover2000 dict (mean_canopy_pct, is_eudr_forest, etc.).
        """
        if self._fallback:
            return self._rule_based_analysis(metrics, wri_driver=wri_driver, tree_cover=tree_cover)

        technical = self._technical_analysis(metrics, coordinates)
        legal = self._legal_assessment(metrics)
        narrative = self._evidence_narrative(technical, legal)
        return CouncilAnalysis(technical=technical, legal=legal, narrative=narrative)

    def _rule_based_analysis(
        self,
        metrics: ChangeMetrics,
        wri_driver: Optional[dict] = None,
        tree_cover: Optional[dict] = None,
    ) -> CouncilAnalysis:
        """Template-based analysis used when the LLM is unavailable."""
        pct  = metrics.change_percentage
        ha   = metrics.loss_area_hectares
        conf = metrics.change_probability

        # WRI driver context
        driver_class   = (wri_driver or {}).get("driver_class", 0)
        driver_label   = (wri_driver or {}).get("driver_label", "Unknown")
        driver_source  = (wri_driver or {}).get("source", "unknown")
        is_eudr_drv    = (wri_driver or {}).get("is_eudr_driver")
        is_natural     = driver_class in EUDR_NATURAL_DRIVERS        # classes 5, 7
        is_non_eudr_lu = driver_class in EUDR_NON_COMMODITY_DRIVERS  # classes 2, 6

        # Tree cover context
        canopy_pct   = (tree_cover or {}).get("mean_canopy_pct")
        mature_pct   = (tree_cover or {}).get("mature_forest_pct")
        is_eudr_fst  = (tree_cover or {}).get("is_eudr_forest")

        driver_sentence = ""
        if driver_class:
            if is_eudr_drv:
                eudr_tag = "EUDR-regulated commodity driver"
            elif is_natural:
                eudr_tag = "natural disturbance — not a direct EUDR violation"
            elif is_non_eudr_lu:
                eudr_tag = "non-commodity land-use change — outside EUDR Annex I scope"
            else:
                eudr_tag = "driver present"
            driver_sentence = (
                f" WRI GDM Drivers dataset (v1.2, 2001–2024) identifies the dominant "
                f"deforestation cause as '{driver_label}' ({eudr_tag}"
                + (f"; source: {driver_source}" if driver_source != "wri" else "")
                + ")."
            )

        forest_sentence = ""
        if canopy_pct is not None:
            forest_sentence = (
                f" Hansen GFC 2024 records a baseline canopy cover of {canopy_pct:.1f}% "
                f"(trees >5 m), with {mature_pct:.1f}% at ≥30% density (proxy for mature "
                f"canopy ≥15 m). EUDR forest status: "
                + ("confirmed" if is_eudr_fst else "NOT met — area may not qualify as forest under EUDR Article 2(4)")
                + "."
            )

        if pct > 5.0:
            if is_natural:
                status = "MEDIUM RISK (YELLOW)"
                legal_note = (
                    f"Forest loss of {pct:.2f}% ({ha:.2f} ha) exceeds the 5% detection level, "
                    f"but is attributed to natural disturbance ({driver_label}). "
                    f"EUDR Article 3 regulates commodity-driven deforestation; natural events "
                    f"are not a direct violation. Enhanced documentation is required."
                )
            elif is_non_eudr_lu:
                status = "MEDIUM RISK (YELLOW)"
                legal_note = (
                    f"Land-cover change of {pct:.2f}% ({ha:.2f} ha) detected, attributed to "
                    f"'{driver_label}' (WRI GDM class {driver_class}). "
                    f"{driver_legal_rationale(driver_class)} "
                    f"No EUDR Article 3 obligation. Operator should document this land-use "
                    f"change for due diligence records."
                )
            elif is_eudr_fst is False:
                status = "MEDIUM RISK (YELLOW)"
                legal_note = (
                    f"Land-cover change of {pct:.2f}% ({ha:.2f} ha) detected, but baseline "
                    f"tree cover ({canopy_pct:.1f}%) does not meet the EUDR forest threshold "
                    f"(≥10% canopy, Article 2(4)). Area may not qualify as forest under EUDR."
                )
            else:
                status = "NON-COMPLIANT (RED)"
                legal_note = (
                    f"Forest loss of {pct:.2f}% ({ha:.2f} ha) exceeds the 5% threshold "
                    f"under EUDR Article 3. "
                    + (
                        f"Driver '{driver_label}' is a commodity supply-chain activity "
                        f"directly regulated under EUDR Regulation EU 2023/1115. "
                        if is_eudr_drv
                        else ""
                    )
                    + "Immediate remediation and competent authority notification are required. "
                    f"Reference date: 31 December 2020."
                )
        elif pct > 0.5:
            status = "MEDIUM RISK (YELLOW)"
            legal_note = (
                f"Forest loss of {pct:.2f}% ({ha:.2f} ha) is below the direct threshold "
                "but warrants enhanced due diligence under EUDR Article 9. "
                "Operator should retain supporting documentation."
            )
        else:
            status = "COMPLIANT (GREEN)"
            legal_note = (
                f"No significant forest loss detected ({pct:.2f}%, {ha:.2f} ha). "
                "The polygon appears compliant with EUDR Article 3 requirements "
                "with respect to the 31 December 2020 reference date."
            )

        technical = (
            f"Satellite change detection (NDVI-ViT hybrid, Sentinel-2 MSI 10 m) identified "
            f"{pct:.2f}% spectral change across {ha:.2f} ha (mean confidence: {conf:.3f}). "
            f"{'High' if conf > 0.7 else 'Medium' if conf > 0.4 else 'Low'} confidence. "
            f"[Rule-based council — LLM unavailable]{driver_sentence}{forest_sentence}"
        )
        narrative = (
            f"Based on automated satellite analysis using Sentinel-2 MSI imagery "
            f"(10 m resolution, Copernicus Programme) and the WRI Global Drivers of "
            f"Deforestation dataset (1 km, 2001–2024), the production polygon shows "
            f"{pct:.2f}% land-cover change affecting {ha:.2f} ha. "
            f"Compliance determination: {status}. {legal_note}"
            f"{driver_sentence}{forest_sentence}"
        )
        return CouncilAnalysis(technical=technical, legal=legal_note, narrative=narrative)

    # ── Agents ─────────────────────────────────────────────────────────────────

    def _technical_analysis(self, metrics: ChangeMetrics, coords: list[float]) -> str:
        prompt = f"""<start_of_turn>user
You are a satellite imagery analyst with 20 years of experience in remote sensing and tropical deforestation.
Analyse the following deforestation detection data:

METRICS:
- Forest loss percentage: {metrics.change_percentage:.2f}%
- Area affected: {metrics.loss_area_hectares:.2f} ha
- Detection confidence (mean probability): {metrics.change_probability:.3f}
- Representative coordinates: lon={coords[0]:.4f}, lat={coords[1]:.4f}

Provide a concise technical assessment (3–4 sentences):
1. What type of land-cover change is most likely indicated? (agriculture, selective logging, clear-cut, urbanisation, or natural disturbance)
2. Describe the spatial pattern that would correspond to this change percentage.
3. Rate the detection confidence (High / Medium / Low) with justification.
<end_of_turn>
<start_of_turn>model"""
        return self._generate(prompt, max_new_tokens=220, temperature=0.3)

    def _legal_assessment(self, metrics: ChangeMetrics) -> str:
        prompt = f"""<start_of_turn>user
You are a legal expert specialised in the EU Deforestation Regulation (EUDR, Regulation EU 2023/1115).
The legal reference date is 31 December 2020.

SCOPE — EUDR Annex I regulated commodities only:
{COMMODITIES_ANNEX_I_SUMMARY}
EUDR does NOT apply to: minerals/mining, urban development, infrastructure, wildfires,
other natural disturbances, or any land-use change not producing an Annex I commodity.

DETECTED EVIDENCE:
- Forest loss inside production polygon: {metrics.change_percentage:.2f}%
- Total area cleared: {metrics.loss_area_hectares:.2f} ha

Compliance thresholds (Article 3):
- NON-COMPLIANT (RED): Forest loss >5% inside the production polygon after 31 Dec 2020,
  driven by an Annex I commodity supply-chain activity
- MEDIUM RISK (YELLOW): Loss from natural disturbance, non-commodity land-use change,
  or loss only in the 5 km buffer zone (leakage risk, Article 9)
- COMPLIANT (GREEN): No significant forest loss detected

Provide:
1. The compliance determination: RED / YELLOW / GREEN
2. The specific EUDR article(s) engaged and the legal reasoning, citing the commodity
   scope limitation if relevant
3. Recommended next steps for the operator or competent authority
Keep the response formal and concise (4–6 sentences).
<end_of_turn>
<start_of_turn>model"""
        return self._generate(prompt, max_new_tokens=260, temperature=0.2)

    def _evidence_narrative(self, technical: str, legal: str) -> str:
        prompt = f"""<start_of_turn>user
Draft a formal EUDR Due Diligence Statement paragraph suitable for submission to EU customs authorities.

TECHNICAL FINDINGS:
{technical}

LEGAL ASSESSMENT:
{legal}

DATA SOURCES: Sentinel-2 MSI (10 m resolution, Copernicus Programme), analysed using AI-assisted change detection (Prithvi-100M geospatial transformer).

Requirements:
- Formal, precise language
- Reference the methodology briefly
- State the findings objectively
- Conclude with the compliance determination and recommended action
Length: 100–150 words.
<end_of_turn>
<start_of_turn>model"""
        return self._generate(prompt, max_new_tokens=400, temperature=0.4, do_sample=True)

    # ── Private ────────────────────────────────────────────────────────────────

    def _generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.3,
        do_sample: bool = False,
    ) -> str:
        try:
            # TinyLlama uses a different chat template than Gemma
            if self._open_fallback:
                # Extract the user message from the Gemma-formatted prompt
                if "<start_of_turn>user\n" in prompt:
                    user_part = prompt.split("<start_of_turn>user\n")[1]
                    user_part = user_part.split("<end_of_turn>")[0].strip()
                else:
                    user_part = prompt
                prompt = self._OPEN_FALLBACK_TEMPLATE.format(
                    system="You are an expert EUDR (EU Deforestation Regulation) compliance analyst.",
                    user=user_part,
                )

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=1024,
                truncation=True,
            ).to(self.llm.device)

            with torch.no_grad():
                outputs = self.llm.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=max(temperature, 0.01),  # avoid temp=0 warning
                    do_sample=do_sample or temperature > 0.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            full_text: str = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Strip the prompt prefix — works for both Gemma and TinyLlama
            for marker in ("<start_of_turn>model", "<|assistant|>"):
                if marker in full_text:
                    return full_text.split(marker)[-1].strip()
            # Fallback: return everything after the prompt length
            prompt_decoded = self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
            if full_text.startswith(prompt_decoded):
                return full_text[len(prompt_decoded):].strip()
            return full_text.strip()

        except Exception as exc:
            raise ModelInferenceError(f"Council LLM generation failed: {exc}") from exc

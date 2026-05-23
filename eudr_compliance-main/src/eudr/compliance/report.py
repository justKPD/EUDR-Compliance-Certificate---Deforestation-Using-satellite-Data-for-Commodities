"""
PDF Due Diligence Statement generator.

Produces a legally structured PDF compliant with EUDR Article 9 (due diligence
statement requirements), including:

  - Compliance status banner (colour-coded)
  - Quantitative analysis table
  - Technical and legal assessments (from the Council)
  - Satellite imagery diff visualisation
  - Cryptographic evidence hash footer
"""
from __future__ import annotations

import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from eudr.config import settings
from eudr.exceptions import ReportGenerationError
from eudr.schemas import AnalysisResult, ComplianceColor

logger = logging.getLogger(__name__)

# Colour map for the compliance banner (R, G, B floats in [0,1])
_BANNER_COLOURS: dict[ComplianceColor, tuple[float, float, float]] = {
    ComplianceColor.GREEN: (0.18, 0.69, 0.31),
    ComplianceColor.YELLOW: (0.92, 0.71, 0.13),
    ComplianceColor.RED: (0.87, 0.18, 0.18),
}


class ReportGenerator:
    """
    Generates a single-page PDF Due Diligence Statement from an AnalysisResult.
    """

    def generate(
        self,
        result: AnalysisResult,
        diff_image: np.ndarray | None = None,
        output_path: Path | None = None,
    ) -> Path:
        """
        Build the PDF report.

        Parameters
        ----------
        result:
            Full analysis result produced by EUDRComplianceSystem.
        diff_image:
            Optional [H, W, 3] uint8 BGR difference image for the visual section.
        output_path:
            Destination path.  Defaults to ``settings.report_output_dir``.
        """
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.utils import ImageReader
            from reportlab.pdfgen import canvas as pdf_canvas
        except ImportError as exc:
            raise ReportGenerationError(
                "reportlab is required for PDF generation. Install with `pip install reportlab`."
            ) from exc

        if output_path is None:
            timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            output_path = settings.report_output_dir / f"EUDR_Report_{timestamp}.pdf"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        page_w, page_h = letter

        try:
            c = pdf_canvas.Canvas(str(output_path), pagesize=letter)
            self._draw_header(c, page_w, page_h, result)
            self._draw_banner(c, page_h, result)
            self._draw_metrics(c, page_h, result)
            self._draw_technical(c, page_h, result)
            self._draw_legal(c, page_h, result)
            if diff_image is not None:
                self._draw_imagery(c, diff_image)
            self._draw_footer(c, result)
            c.save()
        except Exception as exc:
            raise ReportGenerationError(f"PDF generation failed: {exc}") from exc

        logger.info("PDF report written to %s", output_path)
        return output_path

    # ── Section renderers ──────────────────────────────────────────────────────

    def _draw_header(self, c, page_w: float, page_h: float, result: AnalysisResult) -> None:
        c.setFont("Helvetica-Bold", 18)
        c.drawString(50, page_h - 50, "EUDR DUE DILIGENCE STATEMENT")
        c.setFont("Helvetica", 9)
        ts = result.timestamp.strftime("%Y-%m-%d %H:%M UTC")
        c.drawString(50, page_h - 68, f"Generated: {ts}")
        c.drawString(50, page_h - 82, f"Evidence Hash: {result.evidence_hash}")
        c.drawString(50, page_h - 96, f"EUDR Reference Date: {settings.eudr_cutoff_date}")
        # Horizontal rule
        c.setLineWidth(0.5)
        c.line(50, page_h - 104, page_w - 50, page_h - 104)

    def _draw_banner(self, c, page_h: float, result: AnalysisResult) -> None:
        colour = result.compliance.color
        r, g, b = _BANNER_COLOURS[colour]
        c.setFillColorRGB(r, g, b)
        c.setStrokeColorRGB(0, 0, 0)
        c.roundRect(50, page_h - 160, 220, 44, 5, fill=1)
        c.setFillColorRGB(1, 1, 1)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(65, page_h - 131, result.compliance.status.value)
        c.setFillColorRGB(0, 0, 0)
        c.setFont("Helvetica", 9)
        # Wrap reason text next to the banner
        wrapped = self._wrap_text(result.compliance.reason, max_chars=72)
        y = page_h - 120
        for line in wrapped:
            c.drawString(285, y, line)
            y -= 13

    def _draw_metrics(self, c, page_h: float, result: AnalysisResult) -> None:
        y = page_h - 182
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "Quantitative Analysis")
        c.setFont("Helvetica", 10)
        rows = [
            ("Forest loss inside polygon", f"{result.metrics.change_percentage:.2f}%"),
            ("Area affected", f"{result.metrics.loss_area_hectares:.2f} ha"),
            ("Detection confidence (mean prob.)", f"{result.metrics.change_probability:.3f}"),
        ]
        if result.buffer_metrics:
            rows.append(
                ("Forest loss in 5 km buffer zone", f"{result.buffer_metrics.change_percentage:.2f}%")
            )
        for label, value in rows:
            y -= 16
            c.drawString(60, y, f"• {label}:")
            c.drawString(340, y, value)

    def _draw_technical(self, c, page_h: float, result: AnalysisResult) -> None:
        y = page_h - 305
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "Technical Analysis")
        y -= 4
        c.setFont("Helvetica", 9)
        for line in self._wrap_text(result.council.technical, max_chars=100):
            y -= 13
            c.drawString(60, y, line)

    def _draw_legal(self, c, page_h: float, result: AnalysisResult) -> None:
        y = page_h - 420
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "Legal Assessment (EUDR Regulation EU 2023/1115)")
        y -= 4
        c.setFont("Helvetica", 9)
        for line in self._wrap_text(result.council.legal, max_chars=100):
            y -= 13
            c.drawString(60, y, line)

    def _draw_imagery(self, c, diff_image: np.ndarray) -> None:
        """Embed the change-detection visualisation as a PNG image."""
        try:
            from reportlab.lib.utils import ImageReader
            import io
            from PIL import Image as PILImage

            # Convert BGR (OpenCV) → RGB
            rgb = cv2.cvtColor(diff_image, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(rgb)
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            buf.seek(0)
            c.drawImage(ImageReader(buf), 50, 90, width=510, height=160)
            c.setFont("Helvetica", 8)
            c.drawString(50, 80, "Left: 2020 FIR (Vegetation=Red)  |  Centre: NDVI Change (Red=Loss)  |  Right: 2024 FIR")
        except Exception as exc:
            logger.warning("Could not embed imagery in PDF: %s", exc)
            c.setFont("Helvetica-Oblique", 9)
            c.drawString(50, 150, "[Satellite imagery visualisation not available]")

    def _draw_footer(self, c, result: AnalysisResult) -> None:
        c.setFont("Helvetica", 7)
        c.setFillColorRGB(0.4, 0.4, 0.4)
        c.drawString(
            50,
            30,
            "Generated by AI-assisted satellite analysis in accordance with EUDR Article 9. "
            "Satellite sources: Sentinel-2 MSI (Copernicus EU, 10 m).",
        )
        c.drawString(
            50,
            20,
            f"System: EUDR Intelligence v1.0  |  "
            f"Model: Prithvi-100M (NASA-IBM) + Gemma (Google DeepMind)  |  "
            f"Hash: {result.evidence_hash}",
        )

    # ── Utilities ──────────────────────────────────────────────────────────────

    @staticmethod
    def _wrap_text(text: str, max_chars: int = 90) -> list[str]:
        """Simple word-wrap for ReportLab text rendering."""
        words = text.split()
        lines: list[str] = []
        current = ""
        for word in words:
            if len(current) + len(word) + 1 <= max_chars:
                current = f"{current} {word}".lstrip()
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines

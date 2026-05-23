"""
EUDR Regulatory Reference — Regulation (EU) 2023/1115

Single authoritative source for all regulatory constants used across the
compliance engine, council, and report generator.  Any future regulatory
amendment (e.g. updated country risk tiers, extended Annex I) should be
made here and nowhere else.

Official text:
  https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32023R1115

Structure
---------
1.  Regulation metadata
2.  Key article text excerpts
3.  Regulated commodities & products (Annex I)
4.  Forest definition threshold (Article 2(4))
5.  Country risk benchmark tiers (Article 29)
6.  WRI GDM driver → EUDR applicability mapping
7.  Helper functions (is_eudr_relevant_driver, get_country_risk, …)
"""
from __future__ import annotations

from typing import Optional

# ── 1. Regulation metadata ─────────────────────────────────────────────────────

REGULATION_REF: str = "Regulation (EU) 2023/1115"
REGULATION_FULL_TITLE: str = (
    "Regulation (EU) 2023/1115 of the European Parliament and of the Council "
    "of 31 May 2023 on the making available on the Union market and the export "
    "from the Union of certain commodities and products associated with "
    "deforestation and forest degradation and repealing Regulation (EU) No 995/2010"
)
OJ_REFERENCE: str = "OJ L 150, 9.6.2023, p. 206–247"

# EUDR reference date: deforestation occurring AFTER this date triggers non-compliance.
# Source: Article 3(1)(a) — "deforestation-free" means no deforestation after this date.
REFERENCE_DATE: str = "2020-12-31"

# Default detection thresholds
DIRECT_LOSS_THRESHOLD_PCT: float = 5.0    # % loss inside polygon → NON-COMPLIANT
BUFFER_LOSS_THRESHOLD_PCT: float = 20.0   # % loss in 5 km buffer → MEDIUM RISK


# ── 2. Key article text excerpts ───────────────────────────────────────────────

ARTICLE_2_DEFINITIONS: dict[str, str] = {
    "deforestation": (
        "Article 2(3) — 'deforestation' means the conversion of forest to "
        "agricultural use, whether human-induced or not."
    ),
    "forest": (
        "Article 2(4) — 'forest' means land spanning more than 0.5 ha with "
        "trees higher than 5 m and a canopy cover of more than 10%, or trees "
        "able to reach those thresholds in situ, excluding land that is "
        "predominantly under agricultural or urban land use. This definition "
        "aligns with the FAO Forest Resource Assessment definition."
    ),
    "forest_degradation": (
        "Article 2(5) — 'forest degradation' means structural changes to forest "
        "cover in the form of the conversion of (i) primary forests or naturally "
        "regenerating forests into plantation forests or into other wooded land; "
        "or (ii) primary forests into planted forests."
    ),
    "deforestation_free": (
        "Article 2(13) — 'deforestation-free' means that the relevant products "
        "contain, have been fed with or have been made using relevant commodities "
        "produced on land that has not been subject to deforestation after "
        "31 December 2020."
    ),
    "operator": (
        "Article 2(15) — 'operator' means any natural or legal person who, in "
        "the course of a commercial activity, places relevant products on the "
        "Union market or exports them from the Union."
    ),
    "trader": (
        "Article 2(16) — 'trader' means any person in the supply chain other than "
        "the operator who makes relevant products available on the Union market in "
        "the course of a commercial activity."
    ),
    "due_diligence": (
        "Article 2(25) — 'due diligence' means the framework of procedures "
        "implemented by an operator to minimise the risk of relevant products that "
        "do not comply with the requirements of this Regulation being placed or made "
        "available on the Union market or exported from the Union."
    ),
}

ARTICLE_3_PROHIBITION: str = (
    "Article 3 — Relevant commodities and products shall only be placed on the Union "
    "market, made available on the Union market, or exported from the Union, where "
    "ALL of the following conditions are satisfied: "
    "(a) they are deforestation-free; "
    "(b) they have been produced in accordance with the relevant legislation of the "
    "country of production; "
    "(c) they are covered by a due diligence statement."
)

ARTICLE_8_DUE_DILIGENCE: str = (
    "Article 8 — Operators shall, before placing relevant products on the market or "
    "before exporting them, exercise due diligence. Due diligence involves: collecting "
    "information, data and documents needed to comply with Article 9 (risk assessment); "
    "carrying out a risk assessment; and taking risk mitigation measures in accordance "
    "with Article 10."
)

ARTICLE_9_RISK_ASSESSMENT: str = (
    "Article 9 — Operators shall carry out a risk assessment to assess whether "
    "relevant products to be placed on the market involve a risk of non-compliance. "
    "The risk assessment shall take into account: presence of forests in the country "
    "or area of production; compliance with applicable legislation of the country of "
    "production; prevalence of deforestation or forest degradation in the country, "
    "region and area of production; information on supply chain operators and traders; "
    "satellite monitoring data."
)

ARTICLE_10_RISK_MITIGATION: str = (
    "Article 10 — Where risk assessment indicates a non-negligible risk of non-compliance, "
    "operators shall not place the relevant product on the market unless they take "
    "adequate risk-mitigation measures. Such measures shall include obtaining additional "
    "information, data and documents; independent surveys and audits; certified systems; "
    "or other verifications."
)

ARTICLE_13_SIMPLIFIED_DD: str = (
    "Article 13 — Simplified due diligence: operators and traders sourcing from "
    "low-risk countries or low-risk country parts need only collect the information "
    "required by Article 9 and do not need to conduct a full risk assessment, "
    "unless there are indications to the contrary."
)

ARTICLE_22_PENALTIES: str = (
    "Article 22 — Member States shall provide for effective, proportionate and "
    "dissuasive penalties for infringements of this Regulation. Penalties shall include "
    "fines of at least 4% of annual turnover in the Union (where applicable), "
    "confiscation of the relevant products, and temporary exclusion from public procurement."
)


# ── 3. Regulated commodities and products (Annex I) ───────────────────────────
# Each commodity entry lists: scientific name, illustrative HS codes, and
# which derived products are in scope.  HS codes are for reference only;
# the definitive list is the official Annex I text.

REGULATED_COMMODITIES: dict[str, dict] = {
    "cattle": {
        "scientific_name": "Bos taurus / Bubalus bubalis (bovine animals)",
        "hs_codes_illustrative": [
            "0102",  # live bovine animals
            "0201", "0202",  # meat of bovine animals, fresh/chilled/frozen
            "0504",  # guts, bladders, stomachs
            "1502",  # fats of bovine animals
            "1601",  # sausages
            "4101",  # raw hides and skins of bovine animals
            "4104",  # tanned/crust hides and skins of bovine
            "4107",  # leather of bovine
            "9401", "9403",  # leather furniture
        ],
        "annex_ref": "Annex I, row 1",
        "regulated_products": (
            "Live bovine animals; beef and veal; bovine offal; bovine fat; "
            "sausages and similar of bovine origin; bovine raw hides and skins; "
            "tanned/crust bovine leather; finished bovine leather; leather goods; "
            "gelatin; collagen."
        ),
    },
    "cocoa": {
        "scientific_name": "Theobroma cacao",
        "hs_codes_illustrative": [
            "1801",  # cocoa beans, whole or broken
            "1802",  # cocoa shells, husks, skins
            "1803",  # cocoa paste
            "1804",  # cocoa butter, fat and oil
            "1805",  # cocoa powder (not sweetened)
            "1806",  # chocolate and other food preparations containing cocoa
        ],
        "annex_ref": "Annex I, row 2",
        "regulated_products": (
            "Cocoa beans (whole or broken); cocoa shells, husks, skins; "
            "cocoa paste; cocoa butter, fat and oil; cocoa powder; "
            "chocolate and preparations containing cocoa."
        ),
    },
    "coffee": {
        "scientific_name": "Coffea (all species)",
        "hs_codes_illustrative": [
            "0901",  # coffee, roasted or not, decaffeinated or not
            "2101.11", "2101.12",  # extracts, essences, concentrates of coffee
        ],
        "annex_ref": "Annex I, row 3",
        "regulated_products": (
            "Coffee, whether or not roasted or decaffeinated; "
            "coffee husks and skins; coffee substitutes containing coffee; "
            "extracts, essences and concentrates of coffee."
        ),
    },
    "palm_oil": {
        "scientific_name": "Elaeis guineensis (and Elaeis oleifera)",
        "hs_codes_illustrative": [
            "1511",  # palm oil and fractions
            "1513.21", "1513.29",  # palm kernel/babassu oil
            "1516.20",  # vegetable fats and oils (partly/wholly hydrogenated)
            "1517.10",  # margarine (excl. liquid)
            "3823.11", "3823.12", "3823.13", "3823.19",  # industrial fatty acids
        ],
        "annex_ref": "Annex I, row 4",
        "regulated_products": (
            "Palm oil and its fractions; palm kernel oil; margarine (palm-based); "
            "industrial fatty acids from palm; palm stearin; palm olein."
        ),
    },
    "soya": {
        "scientific_name": "Glycine max",
        "hs_codes_illustrative": [
            "1201",  # soya beans, broken or not
            "1208.10",  # flour and meal of soya beans
            "1507",  # soya-bean oil and fractions
            "2304",  # oilcake and solid residues from soya
            "2309",  # preparations of a kind used in animal feeding (soya)
        ],
        "annex_ref": "Annex I, row 5",
        "regulated_products": (
            "Soya beans; soya flour and meal; soya-bean oil; "
            "soya cake/solid residues; prepared animal feed containing soya."
        ),
    },
    "wood": {
        "scientific_name": "All woody arboreal species (timber)",
        "hs_codes_illustrative": [
            "4401",  # fuel wood, wood in chips, sawdust
            "4403",  # wood in the rough
            "4406",  # railway or tramway sleepers of wood
            "4407",  # wood sawn or chipped lengthwise
            "4408",  # sheets for veneering
            "4409",  # wood (including strips and friezes for parquet)
            "4410",  # particle board / oriented strand board
            "4411",  # fibreboard of wood
            "4412",  # plywood, veneered panels
            "4413", "4414", "4415", "4416", "4417", "4418",  # worked wood products
            "4419", "4420", "4421",  # wooden articles
            "47",  # pulp of wood
            "48",  # paper and paperboard
            "9401.61", "9401.69",  # wooden seats
            "9403.30", "9403.40", "9403.50", "9403.60", "9403.91",  # wooden furniture
        ],
        "annex_ref": "Annex I, row 6",
        "regulated_products": (
            "Wood fuel; wood charcoal; wood in the rough; sawn/chipped timber; "
            "veneer; plywood; particle board; fibreboard; wooden railway sleepers; "
            "wood pulp; paper and paperboard; printed matter; wooden furniture."
        ),
    },
    "rubber": {
        "scientific_name": "Hevea brasiliensis (and other natural rubber species)",
        "hs_codes_illustrative": [
            "4001",  # natural rubber, balata, gutta-percha
            "4005",  # compounded rubber, unvulcanised
            "4006",  # other forms of unvulcanised rubber
            "4007",  # vulcanised rubber thread and cord
            "4008",  # plates, sheets, strip of vulcanised rubber
            "4010",  # conveyor or transmission belts of vulcanised rubber
            "4011",  # new pneumatic tyres of rubber
            "4012",  # retreaded or used pneumatic tyres
            "4013",  # inner tubes of rubber
            "4015",  # articles of apparel of vulcanised rubber (gloves)
            "4016",  # other articles of vulcanised rubber
            "4017",  # hard rubber (ebonite)
        ],
        "annex_ref": "Annex I, row 7",
        "regulated_products": (
            "Natural rubber (raw); vulcanised rubber thread; rubber plates/sheets; "
            "conveyor belts; pneumatic tyres; inner tubes; rubber gloves; "
            "other articles of vulcanised rubber."
        ),
    },
}

# Convenience frozenset for fast membership tests
REGULATED_COMMODITY_NAMES: frozenset[str] = frozenset(REGULATED_COMMODITIES.keys())

# Common aliases to normalise user-supplied commodity names
_COMMODITY_ALIASES: dict[str, str] = {
    "beef": "cattle",
    "bovine": "cattle",
    "cow": "cattle",
    "leather": "cattle",
    "palm": "palm_oil",
    "cpo": "palm_oil",            # crude palm oil
    "palm kernel": "palm_oil",
    "soy": "soya",
    "soybeans": "soya",
    "soybean": "soya",
    "timber": "wood",
    "lumber": "wood",
    "paper": "wood",
    "pulp": "wood",
    "natural_rubber": "rubber",
    "cacao": "cocoa",
    "chocolate": "cocoa",
}

COMMODITIES_ANNEX_I_SUMMARY: str = (
    "EUDR Annex I regulated commodities: cattle (beef, leather), cocoa, coffee, "
    "palm oil, soya, wood (timber, paper, pulp), rubber. "
    "EUDR does NOT cover: minerals, metals, fish, biofuels, cotton, tobacco, "
    "flowers, or other agricultural products not listed."
)


# ── 4. Forest definition threshold ────────────────────────────────────────────
# Article 2(4) — aligned with FAO Forest Resource Assessment

FOREST_MIN_CANOPY_PCT: int = 10      # ≥10% canopy cover = qualifies as forest
FOREST_MIN_TREE_HEIGHT_M: float = 5.0  # trees capable of reaching >5 m
FOREST_MIN_AREA_HA: float = 0.5      # land spanning >0.5 ha


# ── 5. Country risk benchmark tiers (Article 29 framework) ────────────────────
# The European Commission publishes a delegated regulation classifying countries
# into three tiers.  As of June 2024 the full list had not been finalised;
# the entries below reflect publicly available guidance and Annex A/B drafts.
#
# Tiers:
#   "high"     → Enhanced due diligence + mandatory risk assessment
#   "standard" → Standard due diligence
#   "low"      → Simplified due diligence (Article 13) — no full risk assessment required

COUNTRY_RISK_TIERS: dict[str, str] = {
    # ── High risk — elevated deforestation rates ───────────────────────────────
    # Sub-Saharan Africa — cocoa belt, Congo Basin
    "BEN": "high", "CMR": "high", "CAF": "high", "CIV": "high",
    "COD": "high", "GHA": "high", "GIN": "high", "LBR": "high",
    "MDG": "high", "NGA": "high", "SLE": "high", "TGO": "high",
    "GAB": "high", "GNQ": "high", "COG": "high",
    # South/South-East Asia — palm oil, rubber, soya
    "IDN": "high", "KHM": "high", "LAO": "high", "MMR": "high",
    "MYS": "high", "PHL": "high", "PNG": "high", "THA": "high",
    "VNM": "high",
    # South America — cattle, soya, cocoa
    "BOL": "high", "BRA": "high", "COL": "high", "ECU": "high",
    "GUY": "high", "PER": "high", "PRY": "high", "SUR": "high",
    "VEN": "high",
    # Central America
    "GTM": "high", "HND": "high", "NIC": "high",
    # ── Low risk — EU Member States + strong governance ────────────────────────
    "AUT": "low", "BEL": "low", "BGR": "low", "HRV": "low",
    "CYP": "low", "CZE": "low", "DNK": "low", "EST": "low",
    "FIN": "low", "FRA": "low", "DEU": "low", "GRC": "low",
    "HUN": "low", "IRL": "low", "ITA": "low", "LVA": "low",
    "LTU": "low", "LUX": "low", "MLT": "low", "NLD": "low",
    "POL": "low", "PRT": "low", "ROU": "low", "SVK": "low",
    "SVN": "low", "ESP": "low", "SWE": "low",
    # Additional low-risk jurisdictions
    "NOR": "low", "CHE": "low", "GBR": "low", "ISL": "low",
    "CAN": "low", "USA": "low", "AUS": "low", "NZL": "low",
    "JPN": "low", "KOR": "low",
}

_DEFAULT_COUNTRY_RISK: str = "standard"


def get_country_risk(iso3: str) -> str:
    """
    Return the EUDR risk tier for an ISO-3166 alpha-3 country code.

    Returns "high", "standard", or "low".
    Unknown codes default to "standard" (conservative).
    """
    return COUNTRY_RISK_TIERS.get(iso3.upper(), _DEFAULT_COUNTRY_RISK)


# ── 6. WRI GDM driver → EUDR applicability ────────────────────────────────────
# Source: WRI Global Drivers of Deforestation 1 km v1.2 (2001–2024)
#   https://developers.google.com/earth-engine/datasets/catalog/
#   projects_landandcarbon_assets_wri_gdm_drivers_forest_loss_1km_v1_2_2001_2024
#
# EUDR Article 3 applies only to commodity supply-chain deforestation.
# Natural disturbances and non-commodity land-use changes are outside its scope.

WRI_DRIVER_EUDR_MAP: dict[int, dict] = {
    0: {
        "label":            "Unknown",
        "is_eudr_relevant": None,
        # None = cannot determine; precautionary principle applies (EUDR Article 3)
        "compliance_disposition": "RED (precautionary)",
        "legal_rationale": (
            "Driver of deforestation could not be determined. "
            "EUDR Article 3 precautionary principle requires treating the operator "
            "as non-compliant until evidence to the contrary is provided."
        ),
    },
    1: {
        "label":            "Permanent agriculture",
        "is_eudr_relevant": True,
        "compliance_disposition": "RED",
        "legal_rationale": (
            "Permanent agricultural expansion (pasture, cropland) is the dominant "
            "driver of EUDR-regulated deforestation. Covers cattle ranching, soya, "
            "palm oil, cocoa, and coffee production — all Annex I commodities."
        ),
    },
    2: {
        "label":            "Hard commodities (mining/extractives)",
        "is_eudr_relevant": False,
        "compliance_disposition": "YELLOW",
        "legal_rationale": (
            "Minerals, metals and extractive industry products are NOT listed in "
            "EUDR Annex I. Land-cover change driven by mining or quarrying does not "
            "constitute an EUDR Article 3 violation. Operators should document this "
            "land-use change for due diligence records, but no EUDR restriction applies."
        ),
    },
    3: {
        "label":            "Shifting cultivation",
        "is_eudr_relevant": True,
        "compliance_disposition": "RED",
        "legal_rationale": (
            "Shifting cultivation that drives permanent land-use change for commodity "
            "production (cocoa, coffee, subsistence-to-commercial transition) falls "
            "within EUDR Article 3 scope. The Commission guidance confirms that "
            "small-scale farmers are not exempt from EUDR obligations."
        ),
    },
    4: {
        "label":            "Logging",
        "is_eudr_relevant": True,
        "compliance_disposition": "RED",
        "legal_rationale": (
            "Commercial and selective logging for timber, wood products, and rubber "
            "is regulated under EUDR Annex I (wood products, HS chapters 44, 47, 48; "
            "rubber, HS chapter 40). Logging-driven forest loss is a direct EUDR "
            "Article 3 violation unless the land was deforested before 31 Dec 2020."
        ),
    },
    5: {
        "label":            "Wildfire",
        "is_eudr_relevant": False,
        "compliance_disposition": "YELLOW",
        "legal_rationale": (
            "Wildfire is a natural disturbance and does not meet the EUDR Article 2(3) "
            "definition of deforestation (conversion to agricultural use). Forest loss "
            "from wildfire does not constitute an Article 3 violation. Enhanced due "
            "diligence documentation under Article 9 is still recommended."
        ),
    },
    6: {
        "label":            "Settlements and infrastructure",
        "is_eudr_relevant": False,
        "compliance_disposition": "YELLOW",
        "legal_rationale": (
            "Urban expansion, road construction, hospital and building development, "
            "and other settlement/infrastructure activities are NOT regulated under "
            "EUDR Annex I. These land-use changes do not produce any commodity listed "
            "in Annex I and therefore fall entirely outside EUDR Article 3 scope. "
            "No EUDR due diligence obligation arises from infrastructure-driven forest "
            "loss. Operators should retain documentation of the land-use change type."
        ),
    },
    7: {
        "label":            "Other natural disturbances",
        "is_eudr_relevant": False,
        "compliance_disposition": "YELLOW",
        "legal_rationale": (
            "Natural disturbances including bark beetle outbreak, fungal disease, "
            "storm damage, drought-induced die-back, and similar natural events do not "
            "constitute deforestation under EUDR Article 2(3). Such losses are common "
            "in Central European managed forests (particularly Picea abies under "
            "climate stress) and are recorded in the Hansen GFC dataset but are "
            "explicitly outside EUDR scope."
        ),
    },
}

# ── Convenience sets — single source of truth ─────────────────────────────────
# All engine and acquisition modules should import from here rather than
# defining their own driver sets.

#: WRI driver classes that trigger EUDR Article 3 non-compliance (RED)
EUDR_COMMODITY_DRIVERS: frozenset[int] = frozenset({1, 3, 4})

#: WRI driver classes that are definitively natural — exempt from EUDR (YELLOW)
EUDR_NATURAL_DRIVERS: frozenset[int] = frozenset({5, 7})

#: WRI driver classes that are non-commodity land-use changes — outside EUDR scope (YELLOW)
EUDR_NON_COMMODITY_DRIVERS: frozenset[int] = frozenset({2, 6})

#: All driver classes that must NOT trigger RED (i.e. all non-commodity drivers)
EUDR_NON_RED_DRIVERS: frozenset[int] = EUDR_NATURAL_DRIVERS | EUDR_NON_COMMODITY_DRIVERS


# ── 7. Helper functions ────────────────────────────────────────────────────────

def is_eudr_relevant_driver(driver_class: int) -> Optional[bool]:
    """
    Return whether a WRI driver class is EUDR-relevant.

    Returns
    -------
    True   — confirmed EUDR commodity driver (RED path)
    False  — definitively NOT EUDR-relevant (YELLOW path)
    None   — unknown / cannot determine (precautionary RED)
    """
    entry = WRI_DRIVER_EUDR_MAP.get(driver_class)
    if entry is None:
        return None
    return entry["is_eudr_relevant"]


def driver_legal_rationale(driver_class: int) -> str:
    """Return the EUDR legal rationale string for a WRI driver class."""
    entry = WRI_DRIVER_EUDR_MAP.get(driver_class, {})
    return entry.get(
        "legal_rationale",
        "Driver unknown — EUDR Article 3 precautionary principle applies.",
    )


def driver_label(driver_class: int) -> str:
    """Return the human-readable label for a WRI driver class."""
    return WRI_DRIVER_EUDR_MAP.get(driver_class, {}).get("label", "Unknown")


def normalise_commodity(commodity: Optional[str]) -> Optional[str]:
    """
    Normalise a commodity name to the canonical Annex I key.

    Returns None if the commodity is not recognised.
    Returns the canonical key (e.g. "cattle", "soya") if recognised.
    """
    if not commodity:
        return None
    c = commodity.lower().strip().replace(" ", "_").replace("-", "_")
    c = _COMMODITY_ALIASES.get(c, c)
    return c if c in REGULATED_COMMODITY_NAMES else None


def is_commodity_in_scope(commodity: Optional[str]) -> bool:
    """
    Return True if the named commodity is regulated under EUDR Annex I.

    Unknown or None commodity names return True (conservative: assume in scope).
    """
    if not commodity:
        return True   # unknown → assume in scope (precautionary)
    return normalise_commodity(commodity) is not None


# ── 8. JRC EU CropMap V1 (EUCROPMAP) ──────────────────────────────────────────
# Source: JRC/D5/EUCROPMAP/V1 — EU-only, 10 m resolution, 2018 and 2022 epochs.
# GEE catalog: https://developers.google.com/earth-engine/datasets/catalog/JRC_D5_EUCROPMAP_V1
#
# The ``classification`` band provides 25 land-use classes at parcel level.
# Only soya (class 233) is a regulated EUDR Annex I commodity within this dataset.
# All other cropland classes are relevant as evidence of PRE-EXISTING agricultural land
# (i.e. the area was already non-forest before the EUDR reference date 2020-12-31).
#
# EUDR Article 2(13) defines "deforestation-free" as: land not subject to deforestation
# AFTER 31 December 2020.  If EUCROPMAP 2018 confirms the land was already arable,
# any NDVI spectral change detected in 2024 imagery is almost certainly crop rotation —
# not post-2020 forest conversion.

EUCROPMAP_ASSET: str = "JRC/D5/EUCROPMAP/V1"

EUCROPMAP_CLASSES: dict[int, str] = {
    0:   "Unknown / Non-agricultural",
    100: "Artificial land",
    211: "Winter cereals",
    212: "Maize",
    213: "Other cereals",
    220: "Root crops",
    221: "Sugar beet",
    222: "Potato",
    230: "Oil crops",
    231: "Rapeseed",
    232: "Sunflower",
    233: "Soya",            # EUDR Annex I commodity (Glycine max)
    240: "Tobacco",
    250: "Other industrial crops",
    300: "Dry pulses, vegetables and flowers",
    310: "Dry pulses",
    320: "Vegetables",
    330: "Flowers",
    400: "Fodder crops",
    410: "Temporary grasslands",
    420: "Fallow land",
    500: "Permanent crops",
    510: "Orchards",
    520: "Vineyards",
    610: "Rice",
    620: "Cotton",
}

# All arable / cropland classes (i.e. agricultural land, not forest, urban, or unknown).
# Presence of these classes in the 2018 layer is the evidence that the land was already
# agricultural BEFORE the EUDR reference date (31 December 2020).
EUCROPMAP_ARABLE_CLASSES: frozenset[int] = frozenset({
    211, 212, 213,          # cereals
    220, 221, 222,          # root crops
    230, 231, 232, 233,     # oil crops incl. soya
    240, 250,               # tobacco, other industrial
    300, 310, 320, 330,     # pulses, vegetables, flowers
    400, 410, 420,          # fodder, temporary grasslands, fallow
    500, 510, 520,          # permanent crops, orchards, vineyards
    610, 620,               # rice, cotton
})

# The only EUDR Annex I commodity present in EUCROPMAP.
# Soya (Glycine max) = class 233.
EUCROPMAP_EUDR_CLASSES: frozenset[int] = frozenset({233})

# Artificial / built-up land class — confirms the area is settlement/infrastructure,
# reinforcing the WRI class 6 non-EUDR attribution.
EUCROPMAP_ARTIFICIAL_CLASSES: frozenset[int] = frozenset({100})


def eucropmap_label(class_value: Optional[int]) -> str:
    """Return the human-readable label for a EUCROPMAP class value."""
    if class_value is None:
        return "Unknown"
    return EUCROPMAP_CLASSES.get(class_value, f"Class {class_value}")

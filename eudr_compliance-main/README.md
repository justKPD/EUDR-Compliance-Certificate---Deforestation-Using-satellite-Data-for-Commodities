# Osapiens Deforestation Intelligence System
### EUDR Compliance Verification · TUM.ai × Osapiens Makeathon

> **"From satellite coordinates to compliance PDF in under 90 seconds."**

---

## What This System Does

The **EU Deforestation Regulation (EUDR, Regulation EU 2023/1115)** requires operators placing
commodities on the EU market (cocoa, coffee, cattle, soy, palm oil, wood, rubber) to prove their
products did not cause or contribute to deforestation **after December 31 2020**.

This system automates that proof. You supply a farm polygon → the system fetches satellite imagery
from before and after the EUDR cutoff, runs AI change-detection, and produces a legally-structured
**PDF Due Diligence Statement** — all from the command line.

> **Latest evaluation: 10 test cases — 5 global (Côte d'Ivoire, Amazon, Congo Basin, Borneo, Colombia) + 5 European (Romania Maramureș, Poland Białowieża, Spain Extremadura, Germany Bavarian Forest, Romania farmland).**

---

## Project Flow (End-to-End)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  1. INPUT                                                               │
│  GeoJSON polygon file  or  inline lon,lat coordinates                  │
└─────────────────────────┬───────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  2. SATELLITE DATA ACQUISITION  (Google Earth Engine)                   │
│                                                                         │
│  Sentinel-2 MSI (optical, 10 m)  — QA60 cloud-masked                   │
│  ├─ 2020 baseline  – B2, B3, B4, B8A, B11, B12  (EUDR cutoff year)     │
│  └─ 2024 current   – same bands                                         │
│  → Biome-aware dry-season windows for cloud-free, phenologically        │
│    consistent imagery (West Africa Nov–Apr, Amazon Jun–Oct, etc.)       │
│                                                                         │
│  Sentinel-1 SAR (radar, 10 m)  — all-weather backup                     │
│  └─ 2024 current   – VV + VH polarisation                               │
│                                                                         │
│  Hansen Global Forest Change 2024 v1.12 — pixel-level 2000–2024 labels │
│  WRI Drivers of Forest Loss   — cause classification (commodity, fire…) │
│  └─ Falls back to Hansen-based driver inference if asset unavailable    │
└─────────────────────────┬───────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  3. PREPROCESSING  (OpenCV + NumPy)                                     │
│  ├─ NaN / cloud-pixel cleaning (QA60 per-pixel bitmask + % filter)      │
│  ├─ Reflectance normalisation (DN ÷ 10 000 → [0, 1])                   │
│  ├─ Resize to 224 × 224 (Prithvi input requirement)                     │
│  ├─ NDVI calculation at native resolution (NIR – Red) / (NIR + Red)    │
│  └─ 3-panel visualisation (baseline | NDVI heatmap | current)          │
└─────────────────────────┬───────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  4. AI ENGINE  (NDVI-ViT Hybrid + Prithvi-100M / ViT-Base fallback)     │
│  ├─ Backbone: frozen Prithvi-100M (NASA-IBM, 768-dim ViT)               │
│  │   → Falls back to ViT-Base/16 if Prithvi weights unavailable         │
│  ├─ Siamese dual-tower: baseline image ─┐                               │
│  │                      current image  ─┴▶ concatenate features         │
│  ├─ Change-detection CNN head → coarse 14×14 probability map            │
│  ├─ NDVI physics prior (full satellite resolution):                      │
│  │   sigmoid(−NDVI_change × 10 − 1.5) — loss onset at NDVI < −0.08    │
│  ├─ Hybrid blend: 10 % ViT + 90 % NDVI prior at full pixel resolution   │
│  │   → Eliminates near-random ViT scores when backbone not fine-tuned   │
│  └─ Optional LoRA fine-tuned checkpoint for domain-specific accuracy    │
└─────────────────────────┬───────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  5. BUFFER ZONE ANALYSIS  (Shapely + GEE)                               │
│  ├─ 5 km radius buffer polygon computed around production area          │
│  └─ Same change-detection pipeline run on buffer for leakage risk       │
└─────────────────────────┬───────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  6. THE COUNCIL  (Gemma 2B / 7B)                                        │
│  Three specialised prompts to the same LLM:                             │
│  ├─ Agent 1 – Technical Analyst: spectral change interpretation         │
│  ├─ Agent 2 – Legal Expert: EUDR Article 3 compliance assessment        │
│  └─ Agent 3 – Report Writer: formal due-diligence narrative             │
└─────────────────────────┬───────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  7. COMPLIANCE DETERMINATION  (EUDR Article 3 + Annex I)                │
│  ├─ RED    NON-COMPLIANT  – forest loss > 5 % from a commodity/logging  │
│  │                          driver (WRI class 1, 3, 4) or unknown       │
│  │                          driver (precautionary principle)             │
│  ├─ YELLOW MEDIUM RISK    – natural disturbance (wildfire, bark beetle, │
│  │                          WRI class 5/7); non-commodity land-use       │
│  │                          change (mining class 2, settlements/infra   │
│  │                          class 6); non-forest baseline; or buffer    │
│  │                          zone leakage only (> 20 % in 5 km zone)    │
│  └─ GREEN  COMPLIANT      – no detectable deforestation                 │
└─────────────────────────┬───────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  8. PDF DUE DILIGENCE STATEMENT                                         │
│  ├─ Compliance status banner (colour-coded)                             │
│  ├─ Quantitative metrics (% loss, hectares, confidence)                 │
│  ├─ Before / After satellite imagery + change map                       │
│  ├─ Technical + legal council analysis                                  │
│  └─ SHA-256 evidence hash (blockchain-ready audit trail)                │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites — Two Services to Set Up

You need credentials for **two** external services. Run `python setup_auth.py` once and it
guides you through both interactively, storing credentials in `~/.eudr/.env`.

---

### Service 1 — Google Cloud + Earth Engine

#### Step 1.1 — Create a Google Cloud Project

1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Click the project dropdown (top left) → **New Project**
3. Give it a name, e.g. `eudr-deforestation`
4. Note the **Project ID** (shown under the project name, e.g. `eudr-deforestation-123456`)

#### Step 1.2 — Enable the Earth Engine API

1. In the Cloud Console, navigate to **APIs & Services → Library**
2. Search for **"Google Earth Engine API"**
3. Click **Enable**

#### Step 1.3 — Grant Service Usage permissions (IAM)

1. In the Cloud Console, navigate to **IAM & Admin → IAM**
2. Click **+ Grant Access**
3. In the "New principals" box, enter your Google account email
4. In the "Role" dropdown, search for **Service Usage Consumer** and select it
5. Click **Save**

> **Shortcut:** Granting yourself **Owner** or **Editor** on the project covers all necessary permissions.

#### Step 1.4 — Register for Earth Engine Non-Commercial Use

1. Go to [earthengine.google.com/noncommercial](https://earthengine.google.com/noncommercial)
2. Click **Register a Noncommercial or Commercial Cloud project**
3. Select **Unpaid usage → Academia or Research**
4. Select your Cloud project from the dropdown
5. Submit — approval is usually instant for research projects

---

### Service 2 — HuggingFace (for Gemma)

Gemma is a gated model — you must accept its licence before downloading.

1. Create a free account at [huggingface.co](https://huggingface.co)
2. Go to [huggingface.co/google/gemma-2b-it](https://huggingface.co/google/gemma-2b-it)
3. Click **"Expand to review and access"** → Accept the licence
4. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
5. Click **New token** → name it `eudr` → Role: **Read** → Generate
6. Copy the token (starts with `hf_...`)

---

## Quick Start

```bash
# 1. Clone the repository
git clone <repo-url>
cd eudr_compliance

# 2. Create a virtual environment (Python 3.9+ supported)
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# 3. Install dependencies  (GPU recommended, CPU fallback works)
pip install --upgrade pip
pip install -e ".[dev]"

# 4. One-time credential setup
#    Prompts for GEE Project ID + HuggingFace token,
#    saves to ~/.eudr/.env, and authenticates GEE via browser (once only)
python setup_auth.py

# 5. Verify the system is ready
python -m eudr check

# 6. Analyse a farm polygon (CLI)
python -m eudr analyze --polygon farm.geojson

# Or supply coordinates directly (lon,lat pairs, space-separated)
python -m eudr analyze --coords "-5.567,7.234 -5.567,7.244 -5.557,7.244 -5.557,7.234"

# 7. Run the evaluation suite (10 test cases: 5 global + 5 European)
python evaluate_pipeline.py

# 8. Launch the Officer UI  →  http://localhost:8000
eudr-api
```

---

## CLI Reference

```
python -m eudr [--log-level LEVEL] COMMAND [OPTIONS]
```

### `analyze` — analyse a single polygon

```bash
# From a GeoJSON file (Polygon, Feature, or FeatureCollection)
python -m eudr analyze --polygon farm.geojson

# From inline coordinates
python -m eudr analyze --coords "-5.567,7.234 -5.567,7.244 -5.557,7.244 -5.557,7.234"

# Skip PDF generation (faster — prints result only)
python -m eudr analyze --polygon farm.geojson --no-pdf

# Skip 5 km buffer zone analysis
python -m eudr analyze --polygon farm.geojson --no-buffer

# Use a LoRA fine-tuned checkpoint for higher accuracy
python -m eudr analyze --polygon farm.geojson --lora-checkpoint ./checkpoints/my_lora.pt
```

**Exit codes:** `0` = COMPLIANT · `1` = error · `2` = NON-COMPLIANT or MEDIUM RISK

### `batch` — analyse a folder of polygons

```bash
python -m eudr batch --input-dir ./polygons --output-dir ./reports
```

Processes every `.geojson` / `.json` file in `--input-dir`, generates one PDF per polygon,
and writes a `batch_summary.json` to `--output-dir`.

### `check` — verify dependencies and credentials

```bash
python -m eudr check
```

Checks Python version, required packages, environment variables, and live GEE connectivity.
Run this after `setup_auth.py` to confirm everything is working.

---

## EUDR Officer UI

The web interface is designed for EUDR compliance officers. Launch the API server and open your browser:

```bash
eudr-api                        # starts at http://localhost:8000
```

**Workflow:**

1. **Select a commodity** — cocoa, coffee, palm oil, soy, cattle, wood, rubber, or other
2. **Draw a polygon** on the satellite map using the polygon tool (top-left of map),
   or paste GeoJSON coordinates in the input box at the bottom of the map
3. Toggle **PDF report** and **buffer zone check** options
4. Click **Run EUDR Analysis** — the progress bar shows all 6 pipeline steps live
5. Results display as a colour-coded **traffic light**:
   - 🟢 **GREEN — COMPLIANT**: no deforestation detected
   - 🟡 **YELLOW — MEDIUM RISK**: natural disturbance, non-commodity land-use change (infrastructure, mining), non-forest baseline, or buffer-zone leakage
   - 🔴 **RED — NON-COMPLIANT**: commodity-driven deforestation post-2020 detected
6. The panel shows WRI driver badge, Hansen forest cover bar, and three council analysis tabs
   (Technical / Legal / Due Diligence narrative)
7. **Download PDF** button appears when a report is generated

---

## Running as a REST API

```bash
# Start the FastAPI server
eudr-api
# → Officer UI at http://localhost:8000
# → Swagger docs at http://localhost:8000/docs
# → API at http://localhost:8000/api/v1/verify

# Verify a polygon (with commodity context)
curl -X POST http://localhost:8000/api/v1/verify \
  -H "Content-Type: application/json" \
  -d '{
    "polygon": {
      "type": "Polygon",
      "coordinates": [[
        [-5.567, 7.234], [-5.567, 7.244],
        [-5.557, 7.244], [-5.557, 7.234],
        [-5.567, 7.234]
      ]]
    },
    "generate_pdf": true,
    "commodity": "cocoa"
  }'
```

The response now includes full context for the UI:

```jsonc
{
  "compliance_status": "NON-COMPLIANT",
  "color": "RED",
  "change_percentage": 12.4,
  "loss_area_hectares": 3.21,
  "change_probability": 0.73,
  "reason": "Forest loss of 12.40% detected inside the production polygon…",
  "wri_driver_label": "Permanent agriculture",
  "wri_driver_class": 1,
  "is_eudr_driver": true,
  "wri_driver_source": "wri",
  "mean_canopy_pct": 54.3,
  "mature_forest_pct": 31.2,
  "is_eudr_forest": true,
  "buffer_change_percentage": 8.1,
  "technical_note": "Sentinel-2 change detection identified…",
  "legal_note": "Forest loss of 12.40% exceeds the 5% threshold under EUDR Article 3…",
  "council_narrative": "Based on automated satellite analysis…",
  "commodity": "cocoa",
  "evidence_hash": "a3f9c2e18b4d...",
  "pdf_url": "/api/v1/reports/report_2026-04-16.pdf",
  "timestamp": "2026-04-16T14:32:01Z"
}
```

**API endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/` | Officer UI (web interface) |
| `POST` | `/api/v1/verify` | Analyse a polygon, returns full compliance result |
| `GET` | `/api/v1/reports/{filename}` | Download a generated PDF |
| `GET` | `/api/v1/health` | Health check |
| `GET` | `/docs` | Interactive Swagger UI |

---

## Credentials

Credentials are resolved in this order (highest priority first):

1. **Shell environment variables** — `export GEE_PROJECT_ID=...`
2. **`.env` file** in the current working directory
3. **`~/.eudr/.env`** — user-level store written by `setup_auth.py`
4. **Interactive prompt** — only when running in a terminal (never in CI/API)

To update credentials at any time:
```bash
python setup_auth.py          # re-runs the wizard, updates ~/.eudr/.env
```

To use a service-account JSON key instead of user auth:
```bash
export GEE_SERVICE_ACCOUNT_KEY=/path/to/key.json
```

---

## Project Structure

```
eudr_compliance/
├── README.md
├── pyproject.toml              # dependencies, build config, scripts (Python 3.9+)
├── .env.example                # credential template — copy to .env
├── setup_auth.py               # one-time GEE + HF credential wizard
├── farm.geojson                # sample Côte d'Ivoire cocoa farm polygon
├── evaluate_pipeline.py        # 10-case evaluation suite (5 global + 5 EU) → ./temp/ + JSON
│
├── src/eudr/                   # core library
│   ├── __init__.py             # lazy module exports (no import cascade)
│   ├── __main__.py             # enables `python -m eudr`
│   ├── cli.py                  # analyze / batch / check commands
│   ├── config.py               # pydantic-settings (reads .env files)
│   ├── schemas.py              # shared Pydantic DTOs
│   ├── exceptions.py           # typed domain exceptions
│   ├── credentials.py          # GEE + HF auth resolution
│   ├── pipeline.py             # EUDRComplianceSystem orchestrator
│   ├── regulations.py          # ★ EUDR regulatory knowledge base
│   │                           #   Annex I commodities + HS codes, Article text,
│   │                           #   country risk tiers (60+ countries), WRI driver →
│   │                           #   EUDR applicability map, helper functions
│   │
│   ├── acquisition/
│   │   └── satellite.py        # GEE: Sentinel-1/2 + Hansen 2024 + WRI fetcher
│   │                           # QA60 cloud masking, biome date windows,
│   │                           # adaptive scale (512 px cap)
│   │                           # WRI fallback: returns class 0 (Unknown) only —
│   │                           # never invents a commodity driver from loss alone
│   │
│   ├── preprocessing/
│   │   └── pipeline.py         # normalise, resize, NDVI, 3-panel diff map
│   │
│   ├── models/
│   │   ├── change_detector.py  # NDVI-ViT hybrid (10% ViT + 90% NDVI prior)
│   │   │                       # full-res change map, Prithvi-100M / ViT-Base/16
│   │   ├── council.py          # Gemma 3-agent council + rule-based fallback
│   │   │                       # legal prompt includes full Annex I commodity scope
│   │   └── fine_tuning.py      # LoRA fine-tuning on Hansen GFC labels
│   │
│   └── compliance/
│       ├── engine.py           # RED / YELLOW / GREEN classifier
│       │                       # WRI class 2 (mining) + class 6 (infra) → YELLOW
│       │                       # Hansen override guarded: never fires for natural
│       │                       # disturbance or infrastructure drivers
│       │                       # WRI 1 km resolution caveat in all driver notes
│       └── report.py           # ReportLab PDF generator
│
├── api/
│   ├── main.py                 # FastAPI app + CORS + static file serving
│   └── routes/compliance.py   # POST /verify (full response), GET /reports/{f}, GET /health
│
├── static/
│   └── index.html              # EUDR Officer UI (Leaflet map, polygon draw, results panel)
│
└── tests/
    ├── test_compliance_engine.py   # 17 tests — metrics, classify (all driver classes),
    │                               #             Hansen override guard, hash, buffer
    ├── test_preprocessing.py       # 7 tests — shapes, NDVI, NaN, dtype
    └── test_api.py                 # 6 tests — endpoints, schema, error cases
```

---

## Evaluation Suite

Runs 10 test cases — 5 global + 5 European — and saves satellite imagery to `./temp/`:

```bash
python evaluate_pipeline.py
```

**European test cases:**

| ID | Location | Expected | Reason |
|----|----------|----------|--------|
| EU-01 | Romania — Maramureș | NON-COMPLIANT | Post-2020 timber logging, forested baseline |
| EU-02 | Poland — Białowieża | COMPLIANT | UNESCO primeval forest, no logging |
| EU-03 | Spain — Extremadura | COMPLIANT | Pre-2020 cleared farmland (EUDR-exempt) |
| EU-04 | Germany — Bavarian Forest | MEDIUM RISK | Bark-beetle outbreak (natural disturbance, not EUDR) |
| EU-05 | Romania — Bucharest farmland | COMPLIANT | No forest baseline, non-EUDR land per Article 2(4) |

---

## Running Tests

No GPU or network required — the ML models are mocked.

```bash
pytest tests/ -v
```

All 33 tests should pass on a plain Python installation with NumPy, OpenCV, Shapely, Pydantic, and FastAPI.

---

## Compliance Thresholds (EUDR Article 3)

| Metric | Default | Configure via `.env` |
|--------|---------|---------------------|
| Reference (cutoff) date | 2020-12-31 | `EUDR_CUTOFF_DATE` |
| Direct loss threshold → RED | > 5 % inside polygon | `DIRECT_LOSS_THRESHOLD` |
| Buffer loss threshold → YELLOW | > 20 % in 5 km zone | `BUFFER_LOSS_THRESHOLD` |
| Buffer radius | 5 km | `BUFFER_RADIUS_KM` |
| Change-detection pixel threshold | 0.35 | hardcoded in `engine.py` |

### WRI Driver Classification → Compliance Path

| WRI Class | Driver | EUDR-Relevant | Determination |
|-----------|--------|---------------|---------------|
| 0 | Unknown | Unknown | RED (precautionary — Article 3) |
| 1 | Permanent agriculture | ✅ Yes | RED |
| 2 | Hard commodities / mining | ❌ No | YELLOW (not Annex I) |
| 3 | Shifting cultivation | ✅ Yes | RED |
| 4 | Logging | ✅ Yes | RED |
| 5 | Wildfire | ❌ No | YELLOW (natural disturbance) |
| 6 | Settlements & infrastructure | ❌ No | YELLOW (not Annex I) |
| 7 | Other natural disturbances | ❌ No | YELLOW (natural disturbance) |

EUDR Annex I covers exactly seven commodities: **cattle, cocoa, coffee, palm oil, soya, wood, rubber**.
Mining, urban development, infrastructure, wildfires, and bark beetle die-off are all outside scope.

---

## AI Change Detection — How It Works

The change-detection pipeline uses a **physics-grounded NDVI–ViT hybrid** rather than relying solely on a neural network:

### Why not pure ViT?

The Prithvi-100M backbone expects 6-band Sentinel-2 data and is frozen at inference time.
When Prithvi is unavailable, the system falls back to ViT-Base/16 — which was trained on
RGB images and produces near-random (~0.5) confidence scores on 6-band satellite data.

### The NDVI Physics Prior

NDVI (Normalised Difference Vegetation Index) is a direct spectral measure of green vegetation:

```
NDVI = (NIR − Red) / (NIR + Red)    ∈ [−1, 1]
```

A significant **drop** in NDVI between the 2020 baseline and the 2024 current image indicates
vegetation loss. The prior converts this physical signal to a loss probability:

```
prior(pixel) = sigmoid(−NDVI_change × 10 − 1.5)
```

| NDVI change | Probability | Interpretation |
|-------------|-------------|----------------|
| −0.5 | 0.95 | Strong forest loss |
| −0.10 | 0.39 | Loss onset — just above 0.35 threshold |
| −0.05 | 0.29 | Seasonal noise — below threshold |
|  0.00 | 0.18 | No change |
| +0.13 | 0.10 | Vegetation gain / healthy crop |

### Hybrid Blend

```
change_map = 0.10 × ViT_output + 0.90 × NDVI_prior
```

- The NDVI prior is computed at **full satellite resolution** (e.g. 112×112)
- The coarse ViT output (14×14) is **upsampled** to match, preserving per-pixel statistics
- At full resolution the 5% EUDR threshold can be precisely measured (12,544 pixels vs 196)
- The 10% ViT weight adds directional information without contaminating the physics signal

### Pre-2020 vs Post-2020 — How the EUDR Cutoff Is Handled

EUDR only regulates deforestation **after December 31 2020**. The pipeline handles this automatically:

- The **baseline imagery is always from 2020** (the EUDR reference year)
- The **current imagery is always from 2024**
- Land cleared **before 2020** already appears as low-NDVI (bare/crop) in the 2020 baseline
  → 2020-to-2024 NDVI change ≈ 0 → **COMPLIANT** (no post-2020 deforestation)
- Land cleared **after 2020** shows high NDVI in the 2020 baseline (forest was present)
  and low NDVI in 2024 → large negative NDVI change → **NON-COMPLIANT**
- **Natural disturbances** (wildfire, bark beetle — WRI driver class 5 or 7) and
  **non-commodity land use** (infrastructure, mining — WRI class 6 or 2) are classified
  **MEDIUM RISK (YELLOW)**, not RED — they fall outside EUDR Annex I scope
- The **Hansen GFC override** (which corroborates NDVI-detected loss with satellite ground-truth)
  is guarded: it only escalates to RED for commodity-relevant or unknown drivers.
  It is explicitly suppressed for natural disturbances and infrastructure — so a bark beetle
  outbreak or hospital construction site in Germany will never be escalated to RED by the
  Hansen signal alone

### Biome-Aware Date Windows

Imagery is acquired during each biome's optimal season for cloud-free, phenologically consistent comparisons:

| Biome | Window | Coverage |
|-------|--------|----------|
| West Africa | November – April | Côte d'Ivoire cocoa belt |
| Amazon | June – October | Pará, Mato Grosso deforestation arc |
| Congo Basin | June – September | DRC, ROC intact forest |
| Borneo | April – July | Kalimantan palm oil frontier |
| Andes highlands | December – March | Colombian/Peruvian coffee regions |
| **Temperate Europe** | **June – August** | Romania, Poland, Spain, Germany, France |

---

## Performance (T4 GPU)

| Stage | Time | GPU RAM |
|-------|------|---------|
| Satellite data acquisition | 30–60 s | — (CPU) |
| Preprocessing | 2–3 s | 0.5 GB |
| Prithvi-100M inference | 4–6 s | 8 GB |
| Gemma Council (3 agents) | 12–20 s | 5.5 GB |
| PDF generation | 2 s | — (CPU) |
| **Total per polygon** | **~55–90 s** | **~14 GB** |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `GEE_PROJECT_ID is not set` | Run `python setup_auth.py` or `export GEE_PROJECT_ID=your-project` |
| `GEE authentication failed` | Run `python setup_auth.py` — uses `ee.Authenticate()` directly (not the CLI, which breaks on Python 3.9) |
| `403 EarthEngine: not signed up` | Complete Step 1.4 — register the project at earthengine.google.com/noncommercial |
| `401 HuggingFace: gated model` | Accept the Gemma licence at huggingface.co/google/gemma-2b-it. System falls back to rule-based analysis automatically. |
| `No module named 'ee'` | `pip install earthengine-api` |
| `No module named 'transformers'` | `pip install transformers accelerate` |
| `CUDA out of memory` | Switch to `gemma-2b-it` (set `COUNCIL_MODEL_ID=google/gemma-2b-it` in `.env`) |
| `No Sentinel-2 scenes` | Cloud cover too high for the date range — system retries at 40% automatically. Biome windows reduce this risk. |
| `GEE request size > 50 MB` | The system auto-scales pixel resolution for large polygons (buffer zones). Scale cap is 512 px per side. |
| `PDF not generated` | `pip install reportlab` |
| `python -m eudr: No module named eudr` | Run `pip install -e .` from the `eudr_compliance/` directory |
| `HF token input not visible` | `setup_auth.py` uses plain `input()` — type the token and press Enter even if nothing appears in the VSCode terminal |
| `Hansen/WRI asset not found` | GEE asset access varies by project. System falls back to class 0 (Unknown driver) — the compliance engine applies the precautionary principle. To enable WRI, ensure your GEE project has read access to `projects/landandcarbon/assets/wri_gdm_drivers_forest_loss_1km_v1_2_2001_2024`. |
| Low confidence scores (~0.5) | Expected when Prithvi-100M is unavailable. The NDVI-ViT hybrid absorbs 90% of the signal from physics rather than the ViT backbone. |

---

## EUDR Regulatory Knowledge Base

All regulatory constants, article definitions, and commodity data live in a single module —
[`src/eudr/regulations.py`](src/eudr/regulations.py) — so any future regulatory change
is made in one place and automatically propagates to the engine, council, and PDF.

### What it contains

**Annex I — Regulated commodities and products**

| Commodity | Scientific name | Illustrative derived products |
|-----------|----------------|-------------------------------|
| Cattle | *Bos taurus / Bubalus bubalis* | Beef, veal, leather, gelatin, tallow |
| Cocoa | *Theobroma cacao* | Cocoa paste/butter/powder, chocolate |
| Coffee | *Coffea* spp. | Roasted/decaffeinated, extracts |
| Palm oil | *Elaeis guineensis* | Palm oil, fractions, margarine |
| Soya | *Glycine max* | Soya meal/oil, animal feed |
| Wood | All arboreal species | Timber, plywood, pulp, paper, furniture |
| Rubber | *Hevea brasiliensis* | Natural rubber, tyres, vulcanised articles |

**Country risk benchmark (Article 29)**

60+ countries pre-classified into `high`, `standard`, or `low` risk tiers. All EU Member States,
Norway, Switzerland, UK, Canada, US, Australia, Japan, and South Korea are `low` risk (simplified
due diligence applies per Article 13). Major deforestation-risk countries (Brazil, Indonesia,
DRC, Côte d'Ivoire, Cambodia, etc.) are `high` risk.

**Key EUDR articles**

| Article | Subject |
|---------|---------|
| Article 2(3) | Definition of deforestation (conversion to agricultural use) |
| Article 2(4) | Definition of forest (>0.5 ha, trees >5 m, ≥10% canopy) |
| Article 3 | Core prohibition — commodities must be deforestation-free, legal, and covered by a due diligence statement |
| Article 8 | Due diligence framework (collect, assess, mitigate) |
| Article 9 | Risk assessment criteria |
| Article 10 | Risk mitigation measures |
| Article 13 | Simplified due diligence for low-risk countries |
| Article 22 | Penalties (≥ 4% of annual EU turnover) |

**Helper functions available for import:**

```python
from eudr.regulations import (
    get_country_risk,          # get_country_risk("DEU") → "low"
    is_eudr_relevant_driver,   # is_eudr_relevant_driver(6) → False
    driver_legal_rationale,    # returns the Article-level legal text for a WRI class
    is_commodity_in_scope,     # is_commodity_in_scope("beef") → True
    normalise_commodity,       # normalise_commodity("soy") → "soya"
    REGULATED_COMMODITY_NAMES, # frozenset of all Annex I canonical names
    EUDR_COMMODITY_DRIVERS,    # frozenset({1, 3, 4})
    EUDR_NATURAL_DRIVERS,      # frozenset({5, 7})
    EUDR_NON_COMMODITY_DRIVERS,# frozenset({2, 6})
)
```

---

## References

- EUDR Regulation (EU) 2023/1115: [eur-lex.europa.eu/eli/reg/2023/1115](https://eur-lex.europa.eu/eli/reg/2023/1115)
- Prithvi-100M (NASA-IBM): [huggingface.co/ibm-nasa-geospatial/Prithvi-100M](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M)
- Hansen Global Forest Change 2024 v1.12: GEE asset `UMD/hansen/global_forest_change_2024_v1_12`
- WRI Global Drivers of Deforestation: GEE asset `projects/landandcarbon/assets/wri_gdm_drivers_forest_loss_1km_v1_2_2001_2024`
- Sentinel-2 QA60 cloud masking: [developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED)
- Google Earth Engine Python API: [developers.google.com/earth-engine/guides/python_install](https://developers.google.com/earth-engine/guides/python_install)

#  Deforestation Intelligence System
### EUDR Compliance Verification · 

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

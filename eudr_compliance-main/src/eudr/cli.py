"""
EUDR Compliance System — standalone CLI.

Usage
-----
    # Analyse a polygon from a GeoJSON file
    python -m eudr analyze --polygon farm.geojson

    # Analyse from inline coordinates (lon lat pairs)
    python -m eudr analyze --coords "-5.567,7.234 -5.567,7.244 -5.557,7.244 -5.557,7.234"

    # Skip PDF, only print compliance status
    python -m eudr analyze --polygon farm.geojson --no-pdf

    # Batch analyse a folder of GeoJSON files
    python -m eudr batch --input-dir ./polygons --output-dir ./reports

    # Check authentication and connectivity
    python -m eudr check

Run ``python -m eudr --help`` for full options.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


# ── Entry point ────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    _configure_logging(args.log_level)

    if args.command == "analyze":
        return _cmd_analyze(args)
    if args.command == "batch":
        return _cmd_batch(args)
    if args.command == "check":
        return _cmd_check(args)

    parser.print_help()
    return 1


# ── Commands ───────────────────────────────────────────────────────────────────

def _cmd_analyze(args: argparse.Namespace) -> int:
    """Analyse a single polygon."""
    polygon = _load_polygon(args)
    if polygon is None:
        return 1

    from eudr.credentials import setup
    from eudr.pipeline import EUDRComplianceSystem
    from eudr.schemas import GeoJSONPolygon

    try:
        setup(interactive=True)
    except RuntimeError as exc:
        print(f"\n[ERROR] {exc}")
        return 1

    lora = getattr(args, "lora_checkpoint", None)
    system = EUDRComplianceSystem(
        lazy_load=False,
        lora_checkpoint=lora if lora and Path(lora).exists() else None,
    )

    geo_polygon = GeoJSONPolygon(coordinates=polygon["coordinates"])

    print("\n" + "=" * 60)
    print(" OSAPIENS EUDR COMPLIANCE SYSTEM")
    print("=" * 60 + "\n")

    result = system.analyze(
        geo_polygon,
        generate_pdf=not args.no_pdf,
        analyze_buffer=not args.no_buffer,
    )

    _print_result(result)

    if result.pdf_path:
        print(f"\n📄 PDF report: {result.pdf_path}")

    return 0 if result.compliance.status.value == "COMPLIANT" else 2


def _cmd_batch(args: argparse.Namespace) -> int:
    """Analyse all GeoJSON files in a directory."""
    from eudr.credentials import setup
    from eudr.pipeline import EUDRComplianceSystem
    from eudr.schemas import GeoJSONPolygon

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    geojson_files = list(input_dir.glob("*.geojson")) + list(input_dir.glob("*.json"))
    if not geojson_files:
        print(f"No .geojson files found in {input_dir}")
        return 1

    try:
        setup(interactive=True)
    except RuntimeError as exc:
        print(f"\n[ERROR] {exc}")
        return 1

    system = EUDRComplianceSystem(lazy_load=False)
    results_summary: list[dict] = []

    for i, geojson_path in enumerate(geojson_files, 1):
        print(f"\n[{i}/{len(geojson_files)}] Analysing: {geojson_path.name}")
        try:
            data = json.loads(geojson_path.read_text())
            # Accept both bare polygon and FeatureCollection
            if data.get("type") == "FeatureCollection":
                data = data["features"][0]["geometry"]
            elif data.get("type") == "Feature":
                data = data["geometry"]
            polygon = GeoJSONPolygon(coordinates=data["coordinates"])
            result = system.analyze(polygon, generate_pdf=True)
            _print_result(result)
            results_summary.append({
                "file": geojson_path.name,
                "status": result.compliance.status.value,
                "color": result.compliance.color.value,
                "change_pct": result.metrics.change_percentage,
                "loss_ha": result.metrics.loss_area_hectares,
                "pdf": result.pdf_path,
                "hash": result.evidence_hash,
            })
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            results_summary.append({"file": geojson_path.name, "error": str(exc)})

    # Write batch summary
    summary_path = output_dir / "batch_summary.json"
    summary_path.write_text(json.dumps(results_summary, indent=2))
    print(f"\n✅ Batch complete. Summary: {summary_path}")
    _print_batch_summary(results_summary)
    return 0


def _cmd_check(args: argparse.Namespace) -> int:  # noqa: ARG001
    """Check GEE connectivity and HF access."""
    print("\nRunning system check …\n")
    ok = True

    # Python version
    pv = sys.version_info
    status = "✓" if pv >= (3, 9) else "✗"
    print(f"  {status} Python {pv.major}.{pv.minor}.{pv.micro}")
    if pv < (3, 9):
        print("    ↳ Python 3.9+ required")
        ok = False

    # Check key packages
    for pkg in ["ee", "torch", "cv2", "transformers", "reportlab", "shapely"]:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg} — not installed")
            ok = False

    # Check env vars
    print()
    for var in ["GEE_PROJECT_ID", "HF_TOKEN"]:
        val = os.environ.get(var, "")
        if val:
            masked = val[:4] + "…" + val[-4:] if len(val) > 8 else "***"
            print(f"  ✓ {var} = {masked}")
        else:
            # Also check .env file
            from eudr.config import settings
            attr = var.lower()
            cfg_val = getattr(settings, attr, "")
            if cfg_val:
                masked = cfg_val[:4] + "…" if len(cfg_val) > 4 else "***"
                print(f"  ✓ {var} = {masked}  (from .env)")
            else:
                print(f"  ✗ {var} — not set  (add to .env or export in shell)")
                if var == "GEE_PROJECT_ID":
                    ok = False

    # GEE connectivity
    print()
    try:
        from eudr.credentials import _init_gee
        from eudr.config import settings
        _init_gee(settings.gee_project_id)
        print("  ✓ Google Earth Engine — connected")
    except Exception as exc:
        print(f"  ✗ Google Earth Engine — {exc}")
        ok = False

    print()
    if ok:
        print("✅ System check passed. Ready to run.\n")
    else:
        print("❌ System check failed. Fix the issues above, then run `python setup_auth.py`.\n")
    return 0 if ok else 1


# ── Parser ─────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m eudr",
        description="Osapiens EUDR Deforestation Intelligence — standalone CLI",
    )
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # analyze
    p_analyze = sub.add_parser("analyze", help="Analyse a single farm polygon")
    src = p_analyze.add_mutually_exclusive_group(required=True)
    src.add_argument("--polygon", metavar="FILE",
                     help="Path to GeoJSON file containing the polygon")
    src.add_argument("--coords", metavar="'lon,lat ...'",
                     help="Space-separated lon,lat pairs forming the polygon ring")
    p_analyze.add_argument("--no-pdf", action="store_true",
                           help="Skip PDF generation (print results only)")
    p_analyze.add_argument("--no-buffer", action="store_true",
                           help="Skip 5 km buffer zone analysis")
    p_analyze.add_argument("--lora-checkpoint", metavar="PATH",
                           help="Path to a LoRA fine-tuned model checkpoint")

    # batch
    p_batch = sub.add_parser("batch", help="Analyse all GeoJSON files in a directory")
    p_batch.add_argument("--input-dir", required=True, metavar="DIR",
                         help="Directory containing .geojson files")
    p_batch.add_argument("--output-dir", default="./reports", metavar="DIR",
                         help="Directory for PDF reports and summary (default: ./reports)")

    # check
    sub.add_parser("check", help="Verify GEE auth and dependencies")

    return parser


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_polygon(args: argparse.Namespace) -> Optional[dict]:
    if args.polygon:
        path = Path(args.polygon)
        if not path.exists():
            print(f"[ERROR] File not found: {path}")
            return None
        data = json.loads(path.read_text())
        if data.get("type") == "FeatureCollection":
            data = data["features"][0]["geometry"]
        elif data.get("type") == "Feature":
            data = data["geometry"]
        return data

    # --coords mode: "lon,lat lon,lat ..."
    try:
        points = [list(map(float, p.split(","))) for p in args.coords.split()]
        if points[0] != points[-1]:
            points.append(points[0])  # close the ring
        return {"type": "Polygon", "coordinates": [points]}
    except Exception as exc:
        print(f"[ERROR] Invalid coordinates: {exc}")
        return None


def _print_result(result) -> None:
    colour_icon = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}
    icon = colour_icon.get(result.compliance.color.value, "❓")
    print(f"\n{icon}  {result.compliance.status.value}")
    print(f"   {result.compliance.reason}")
    print(f"\n   Forest loss:  {result.metrics.change_percentage:.2f}%")
    print(f"   Area:         {result.metrics.loss_area_hectares:.2f} ha")
    print(f"   Confidence:   {result.metrics.change_probability:.3f}")
    if result.buffer_metrics:
        print(f"   Buffer loss:  {result.buffer_metrics.change_percentage:.2f}%  (5 km zone)")
    print(f"   Evidence hash: {result.evidence_hash}")


def _print_batch_summary(results: list[dict]) -> None:
    counts = {"COMPLIANT": 0, "MEDIUM RISK": 0, "NON-COMPLIANT": 0, "ERROR": 0}
    for r in results:
        if "error" in r:
            counts["ERROR"] += 1
        else:
            counts[r.get("status", "ERROR")] = counts.get(r.get("status", "ERROR"), 0) + 1
    print("\nBatch summary:")
    print(f"  🟢 Compliant:      {counts['COMPLIANT']}")
    print(f"  🟡 Medium risk:    {counts['MEDIUM RISK']}")
    print(f"  🔴 Non-compliant:  {counts['NON-COMPLIANT']}")
    print(f"  ❌ Errors:         {counts['ERROR']}")


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
One-time authentication setup for the EUDR Compliance System.

Run this script once to cache your credentials on disk.
After that, `python -m eudr analyze ...` works without re-authenticating.

What this script does
---------------------
  1. Reads / prompts for GEE_PROJECT_ID and HF_TOKEN
  2. Writes them to ~/.eudr/.env  (user-level credential store)
  3. Authenticates Google Earth Engine using ONE of:
       a) Service-account JSON key  (if GEE_SERVICE_ACCOUNT_KEY is set)
       b) `earthengine authenticate` CLI — opens a browser once, caches to disk
       c) `gcloud auth application-default login` — if you have gcloud installed
  4. Verifies the connection by making a tiny GEE API call

Re-run only if you change credentials or GEE expires (tokens last ~90 days).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


# ── Helpers ────────────────────────────────────────────────────────────────────

def _ask(prompt: str, default: str = "", secret: bool = False) -> str:
    """Prompt the user, returning `default` if they just press Enter."""
    if default:
        display = "[kept] " if secret else f"[{default}] "
    else:
        display = ""
    try:
        val = input(f"{prompt} {display}: ").strip()
    except (KeyboardInterrupt, EOFError):
        print()
        val = ""
    return val or default


def _run(*cmd: str) -> bool:
    """Run a shell command, returning True on success."""
    print(f"\n  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    return result.returncode == 0


def _save_env(path: Path, values: dict[str, str]) -> None:
    """Merge key=value pairs into an .env file without clobbering others."""
    existing: dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                existing[k.strip()] = v.strip()
    existing.update({k: v for k, v in values.items() if v})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# EUDR Compliance — user credentials (managed by setup_auth.py)\n"
        + "\n".join(f"{k}={v}" for k, v in existing.items())
        + "\n"
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 60)
    print("  EUDR Compliance System — First-time Setup")
    print("=" * 60)
    print()
    print("This wizard stores credentials in ~/.eudr/.env")
    print("Press Enter to keep the current value shown in brackets.\n")

    env_path = Path.home() / ".eudr" / ".env"

    # ── Step 1: Load existing values as defaults ───────────────────────────────
    existing: dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                existing[k.strip()] = v.strip()

    cur_project = os.environ.get("GEE_PROJECT_ID", existing.get("GEE_PROJECT_ID", ""))
    cur_hf = os.environ.get("HF_TOKEN", existing.get("HF_TOKEN", ""))
    cur_key = os.environ.get("GEE_SERVICE_ACCOUNT_KEY",
                              existing.get("GEE_SERVICE_ACCOUNT_KEY", ""))

    # ── Step 2: Collect GEE project ID ────────────────────────────────────────
    print("── Google Earth Engine ───────────────────────────────────")
    print("Your GCP project ID (e.g. 'my-gee-project-12345').")
    print("Get it from: https://console.cloud.google.com → Project selector\n")
    gee_project = _ask("GEE_PROJECT_ID", default=cur_project)
    if not gee_project:
        print("\n[ERROR] GEE_PROJECT_ID is required. Aborting.")
        return 1

    print()
    print("(Optional) Path to a service-account JSON key file.")
    print("Leave blank to use Application Default Credentials (browser login).\n")
    gee_key = _ask("GEE_SERVICE_ACCOUNT_KEY (optional)", default=cur_key)

    # ── Step 3: Collect HuggingFace token ─────────────────────────────────────
    print()
    print("── HuggingFace ───────────────────────────────────────────")
    print("Required to download Gemma (gated model). Generate at:")
    print("  https://huggingface.co/settings/tokens\n")
    hf_token = _ask("HF_TOKEN (hf_...)", default=cur_hf, secret=True)

    # ── Step 4: Save to ~/.eudr/.env ──────────────────────────────────────────
    to_save = {
        "GEE_PROJECT_ID": gee_project,
        "HF_TOKEN": hf_token,
        "GEE_SERVICE_ACCOUNT_KEY": gee_key,
    }
    _save_env(env_path, {k: v for k, v in to_save.items() if v})
    print(f"\n  ✓ Credentials saved to {env_path}")

    # ── Step 5: GEE authentication ────────────────────────────────────────────
    print()
    print("── Google Earth Engine Authentication ────────────────────")

    if gee_key and Path(gee_key).exists():
        print(f"  ✓ Service-account key found at {gee_key}")
        print("  ✓ No browser login needed — service accounts are always valid.")
    else:
        print("No service-account key found.")
        print("Opening browser for Google Earth Engine authentication …")
        print("(A browser tab will open — sign in with your Google account and approve access.)\n")
        try:
            import ee  # noqa: PLC0415
            ee.Authenticate()
            print("\n  ✓ Earth Engine authentication complete.")
        except Exception as exc:
            print(f"\n  [WARN] ee.Authenticate() failed: {exc}")
            print("  You can retry authentication manually later with:")
            print("    python setup_auth.py verify")

    # ── Step 6: Verify connectivity ────────────────────────────────────────────
    print()
    print("── Verifying GEE connectivity ────────────────────────────")

    # Set env vars for the verification call
    os.environ["GEE_PROJECT_ID"] = gee_project
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    if gee_key:
        os.environ["GEE_SERVICE_ACCOUNT_KEY"] = gee_key

    try:
        from eudr.credentials import _init_gee
        _init_gee(gee_project)
        print("  ✓ Google Earth Engine — connected successfully!")
    except Exception as exc:
        print(f"\n  [WARN] GEE verification failed: {exc}")
        print("  If you just authenticated, credentials may need a moment to propagate.")
        print("  Run `python -m eudr check` to verify before your first analysis.")

    print()
    print("=" * 60)
    print("  Setup complete!")
    print()
    print("  Next steps:")
    print("    python -m eudr check              # verify everything is ready")
    print("    python -m eudr analyze --polygon farm.geojson")
    print("    python -m eudr --help             # all options")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    # Allow `python setup_auth.py verify` to just run the connectivity check
    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        gee_id = os.environ.get("GEE_PROJECT_ID", "")
        if not gee_id:
            env_path = Path.home() / ".eudr" / ".env"
            if env_path.exists():
                for line in env_path.read_text().splitlines():
                    if line.startswith("GEE_PROJECT_ID="):
                        gee_id = line.split("=", 1)[1].strip()
        if not gee_id:
            print("[ERROR] GEE_PROJECT_ID not found. Run: python setup_auth.py")
            sys.exit(1)
        try:
            from eudr.credentials import _init_gee
            _init_gee(gee_id)
            print("✓ GEE connected.")
            sys.exit(0)
        except Exception as exc:
            print(f"✗ {exc}")
            sys.exit(1)
    else:
        sys.exit(main())

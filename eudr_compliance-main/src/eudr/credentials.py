"""
Credential manager for Google Earth Engine and HuggingFace.

Designed for standalone Python projects — no Colab, no browser widgets.

Resolution priority
-------------------
  1. Explicit keyword arguments to ``setup()``
  2. Environment variables (GEE_PROJECT_ID, HF_TOKEN)
  3. `.env` file in the current directory (via python-dotenv)
  4. `~/.eudr/.env`  — user-level credential store
  5. Interactive getpass prompt (terminal only — skipped in non-TTY environments)

GEE authentication priority
----------------------------
  1. Service-account JSON key  (set GEE_SERVICE_ACCOUNT_KEY=/path/to/key.json)
  2. Application Default Credentials already in place (from `gcloud auth` or
     a prior `earthengine authenticate` run)
  3. If nothing is found → prints clear instructions and raises an error rather
     than blocking forever on `ee.Authenticate()`.

To authenticate GEE for the first time, run:
    python setup_auth.py
"""
from __future__ import annotations

import getpass
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class Credentials:
    gee_project_id: str
    hf_token: str


# ── Public entry point ─────────────────────────────────────────────────────────

def setup(
    gee_project_id: Optional[str] = None,
    hf_token: Optional[str] = None,
    *,
    interactive: bool = True,
) -> Credentials:
    """
    Resolve and apply GEE + HuggingFace credentials.

    Parameters
    ----------
    gee_project_id:
        Explicit override — skips all auto-detection when supplied.
    hf_token:
        Explicit override — skips all auto-detection when supplied.
    interactive:
        Allow terminal prompts if values cannot be found.
        Set to False in automated / CI environments.

    Returns
    -------
    Credentials  (also written to os.environ for downstream config)
    """
    _load_dotenv()

    gee_project_id = gee_project_id or _get("GEE_PROJECT_ID", "GEE Project ID",
                                             interactive=interactive)
    hf_token = hf_token or _get("HF_TOKEN", "HuggingFace token (hf_...)",
                                 secret=True, required=False, interactive=interactive)

    creds = Credentials(gee_project_id=gee_project_id, hf_token=hf_token)
    _apply(creds)
    return creds


# ── Resolution helpers ─────────────────────────────────────────────────────────

def _load_dotenv() -> None:
    """Load .env files without crashing if python-dotenv is absent."""
    try:
        from dotenv import load_dotenv  # type: ignore[import]
        for candidate in [".env", str(Path.home() / ".eudr" / ".env")]:
            if Path(candidate).exists():
                load_dotenv(candidate, override=False)
                log.debug("Loaded %s", candidate)
    except ImportError:
        pass


def _get(
    env_key: str,
    label: str,
    secret: bool = False,
    required: bool = True,
    interactive: bool = True,
) -> str:
    """Return the value for `env_key`, prompting if needed."""
    val = os.environ.get(env_key, "").strip()
    if val:
        log.info("  ✓ %s loaded from environment", label)
        return val

    if not interactive or not _is_tty():
        if required:
            raise RuntimeError(
                f"{env_key} is not set and no interactive terminal is available.\n"
                f"Set it via:  export {env_key}=<value>  or add it to .env"
            )
        return ""

    # Interactive terminal prompt
    prompt_fn = getpass.getpass if secret else input
    try:
        val = prompt_fn(f"Enter {label}: ").strip()
    except (KeyboardInterrupt, EOFError):
        val = ""

    if not val:
        if required:
            raise ValueError(f"{label} cannot be empty.")
        return ""

    os.environ[env_key] = val
    return val


def _is_tty() -> bool:
    return hasattr(sys.stdin, "isatty") and sys.stdin.isatty()


# ── Apply credentials ──────────────────────────────────────────────────────────

def _apply(creds: Credentials) -> None:
    """Propagate credentials into the environment and initialise services."""
    os.environ["GEE_PROJECT_ID"] = creds.gee_project_id
    if creds.hf_token:
        os.environ["HF_TOKEN"] = creds.hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = creds.hf_token

    _init_gee(creds.gee_project_id)
    _login_huggingface(creds.hf_token)


def _init_gee(project_id: str) -> None:
    """
    Initialise Google Earth Engine without launching a browser.

    Auth strategy (in order):
      1. Service-account key file (GEE_SERVICE_ACCOUNT_KEY env var)
      2. Application Default Credentials already cached on disk
         (~/.config/earthengine/credentials  or GOOGLE_APPLICATION_CREDENTIALS)
    If neither works, raise a clear error with setup instructions.
    """
    if not project_id:
        log.warning("GEE_PROJECT_ID not set — Earth Engine will not be initialised.")
        return

    try:
        import ee  # type: ignore[import]
    except ImportError:
        log.warning("earthengine-api not installed — run: pip install earthengine-api")
        return

    # 1. Service-account JSON key
    key_path = os.environ.get("GEE_SERVICE_ACCOUNT_KEY", "").strip()
    if key_path and Path(key_path).exists():
        try:
            creds = ee.ServiceAccountCredentials(email="", key_file=key_path)
            ee.Initialize(credentials=creds, project=project_id)
            print(f"  ✓ Earth Engine initialised (service account, project: {project_id})")
            return
        except Exception as exc:
            log.warning("Service-account auth failed: %s", exc)

    # 2. Cached Application Default Credentials (ADC)
    try:
        ee.Initialize(project=project_id)
        print(f"  ✓ Earth Engine initialised (project: {project_id})")
        return
    except Exception as exc:
        log.debug("ADC init failed: %s", exc)

    # 3. Nothing worked — give clear instructions
    raise RuntimeError(
        f"Could not authenticate with Google Earth Engine (project: {project_id}).\n\n"
        "To authenticate, run ONE of:\n"
        "  python setup_auth.py            # guided first-time setup\n"
        "  earthengine authenticate        # CLI (pip install earthengine-api)\n"
        "  gcloud auth application-default login  # if you have gcloud\n\n"
        "Then re-run your script."
    )


def _login_huggingface(token: str) -> None:
    """Set HuggingFace token for model downloads (non-blocking if not installed)."""
    if not token:
        return
    try:
        from huggingface_hub import login  # type: ignore[import]
        login(token=token, add_to_git_credential=False)
        print("  ✓ HuggingFace authenticated")
    except ImportError:
        log.debug("huggingface_hub not installed — token stored in env only")
    except Exception as exc:
        log.warning("HuggingFace login failed: %s", exc)

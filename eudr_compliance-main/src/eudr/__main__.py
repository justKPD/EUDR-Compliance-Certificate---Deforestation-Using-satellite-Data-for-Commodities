"""Allow `python -m eudr` to invoke the CLI."""
from __future__ import annotations

import sys

from eudr.cli import main

sys.exit(main())

"""Package entry point: ``python -m credit_risk``."""

from __future__ import annotations

import sys

from credit_risk.cli import main

if __name__ == "__main__":
    sys.exit(main())

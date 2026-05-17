"""Top-level CLI shim — runs the same code as `python -m cutaioffical_engine`.

Useful when the package isn't installed editable and you just want to invoke
`python cli.py <video>` from the repo root.
"""
from __future__ import annotations

import sys

from cutaioffical_engine.__main__ import main


if __name__ == "__main__":
    sys.exit(main())

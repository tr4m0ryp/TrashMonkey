"""Test bootstrap: import src/ layout without installation; no network, no GPU."""

import os
import sys
from pathlib import Path

# Disable albumentations' import-time update check (network call).
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

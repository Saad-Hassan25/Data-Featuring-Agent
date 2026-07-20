"""Make the package importable when running the tests without installing."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# keep test output readable; the harness's numbers are unaffected
warnings.filterwarnings("ignore", message="X does not have valid feature names")

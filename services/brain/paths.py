import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("FLY_DATA_DIR", REPO / "data"))
RAW_DIR = DATA_DIR / "raw"
ARTIFACTS_DIR = Path(os.environ.get("FLY_ARTIFACTS_DIR", REPO / "artifacts"))

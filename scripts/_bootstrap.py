"""Make the babymotion package importable when running scripts directly.

Also caps the BLAS thread pool before numpy/cv2 load (see app/__init__):
the perf-critical scripts must import this before any numpy import.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

BABYMOTION_DIR = Path(__file__).resolve().parent.parent
if str(BABYMOTION_DIR) not in sys.path:
    sys.path.insert(0, str(BABYMOTION_DIR))

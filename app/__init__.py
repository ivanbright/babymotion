"""BabyMotion V1 - adaptive digital-twin interaction engine."""

import os

# Cap the BLAS pool before numpy loads: this workload is many tiny per-frame
# matmuls, where the default thread pool costs more in sync overhead than it
# gains (measured ~5x slower twin renders with the default pool).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

__version__ = "0.1.0"

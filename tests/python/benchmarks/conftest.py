import sys
from pathlib import Path

# The benchmark scripts import their siblings as top-level modules, as `python benchmarks/run_benchmark.py` does.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarks"))

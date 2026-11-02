"""Locations of input data and result files (override with DNFR_DATA / DNFR_RESULTS)."""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("DNFR_DATA", os.path.join(ROOT, "data"))
OUT = os.environ.get("DNFR_RESULTS", os.path.join(ROOT, "results"))

"""
paths.py

The necessary paths for Ray.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FED_AVG = os.path.abspath(os.path.join(HERE, "..", "fed_avg"))

for p in (HERE, FED_AVG):
    if p not in sys.path:
        sys.path.append(p)

# Ray workers are separate processes: they need the same import paths.
# This must run before ray.init().
existing = os.environ.get("PYTHONPATH")
os.environ["PYTHONPATH"] = os.pathsep.join([HERE, FED_AVG] + ([existing] if existing else []))
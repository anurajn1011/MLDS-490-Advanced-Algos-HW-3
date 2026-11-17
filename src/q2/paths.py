'''
paths.py

Providing access to modules in the `fed_avg` directory under `q1`.
'''

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FED_AVG = os.path.abspath(os.path.join(HERE, "..", "q1", "fed_avg"))

for p in (HERE, FED_AVG):
    if p not in sys.path:
        sys.path.append(p)
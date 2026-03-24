import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

if sys.version_info < (3, 10):
    sys.stderr.write("Python 3.10+ required\n")
    sys.exit(1)

if len(sys.argv) == 1:
    sys.argv.append("roundtrip")

from frame_pipeline import main

main()

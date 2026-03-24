"""
收端专用入口：等价于 ``python run.py decode|decode-file|from-mp4|extract-mp4``，
但仅挂载解码子命令，默认 ``-h``。

  python run_decode.py png -i frame.png -o out.bin
  python run_decode.py frames -i frames_dir/ -o out.bin
  python run_decode.py mp4 -i video.mp4 -o out.bin
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

if sys.version_info < (3, 10):
    sys.stderr.write("Python 3.10+ required\n")
    sys.exit(1)

if len(sys.argv) == 1:
    sys.argv.append("-h")

from decode_pipeline import main

main()

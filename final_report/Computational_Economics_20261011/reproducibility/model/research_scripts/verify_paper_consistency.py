"""Strict compatibility entry for the stage-1 manuscript/archive verifier."""
from pathlib import Path
import runpy
if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().parents[2]/'scripts/stage1_verify.py'), run_name='__main__')

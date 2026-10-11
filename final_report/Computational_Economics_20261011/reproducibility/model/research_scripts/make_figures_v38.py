"""Regenerate identified stage-1 figures and synchronized editable diagrams."""
from pathlib import Path
import runpy
if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().parents[2]/'scripts/stage1_figures.py'), run_name='__main__')

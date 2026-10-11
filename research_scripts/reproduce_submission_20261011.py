"""Verify or reproduce the frozen 2026-10-11 submission; no simulations."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "final_report/Computational_Economics_20261011/reproducibility"

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify-only", action="store_true", help="Standard-library checksum verification only")
    ap.add_argument("--figures", action="store_true", help="Also regenerate 15 figures after numerical verification")
    args = ap.parse_args()
    subprocess.run([sys.executable, str(KIT / "analysis/verify_package.py")], cwd=KIT, check=True)
    if args.verify_only:
        if args.figures:
            ap.error("--figures requires numerical reproduction")
        return
    subprocess.run([sys.executable, str(KIT / "analysis/reproduce.py")], cwd=KIT, check=True)
    if args.figures:
        subprocess.run([sys.executable, str(KIT / "artwork/build_artwork.py")], cwd=KIT, check=True)

if __name__ == "__main__":
    main()

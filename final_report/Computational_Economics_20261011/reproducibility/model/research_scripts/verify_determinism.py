"""Fixed-seed full-trajectory verifier; missing required fields fail explicitly."""
from verify_seed_reproducibility import main
if __name__ == '__main__':
    raise SystemExit(main())

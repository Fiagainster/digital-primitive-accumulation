"""Verify immutable kit files using SHA-256; generated output is excluded."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]

def main():
    failures=[]
    count=0
    for line in (ROOT/"files.sha256").read_text(encoding="utf-8").splitlines():
        if not line:continue
        expected,name=line.split("  ",1)
        target=ROOT/name
        if target.is_absolute() and not target.resolve().is_relative_to(ROOT):
            raise ValueError("Fingerprint path escapes the kit")
        if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest()!=expected:
            failures.append(name)
        count+=1
    result={"passed":not failures,"checked_files":count,"failures":failures}
    print(json.dumps(result,ensure_ascii=False))
    if failures:raise SystemExit(1)

if __name__=="__main__":main()

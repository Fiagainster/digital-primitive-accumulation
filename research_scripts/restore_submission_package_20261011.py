"""Restore and verify the original delivery from the two repository parts."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

VERSION = "Computational_Economics_20261011"

def digest(data):
    return hashlib.sha256(data).hexdigest()

def member(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Manifest path escapes its repository")
    return path

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paper-root", type=Path, required=True)
    ap.add_argument("--code-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, help="A new or empty directory for the restored package")
    ap.add_argument("--zip-output", type=Path, help="Recreate the byte-identical original ZIP")
    args = ap.parse_args()
    roots = {"paper": args.paper_root.resolve(), "code": args.code_root.resolve()}
    manifest_path = member(roots["paper"], "review/" + VERSION + "/repository_split.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = {}
    for entry in manifest["files"]:
        data = member(roots[entry["repository"]], entry["repository_path"]).read_bytes()
        if digest(data) != entry["sha256"]:
            raise ValueError("Changed or missing delivery file: " + entry["original_path"])
        payload[entry["original_path"]] = data
    # Validate both the restored outer checksum ledger and all nested ZIP CRCs.
    count = 0
    for line in payload["files.sha256"].decode("utf-8").splitlines():
        expected, name = line.split("  ", 1)
        if digest(payload[name]) != expected:
            raise ValueError("Original fingerprint failed: " + name)
        count += 1
    import io
    for name in ("submission/Manuscript_Source.zip", "submission/Supplement_Source.zip", "submission/ESM_2.zip"):
        with zipfile.ZipFile(io.BytesIO(payload[name])) as nested:
            if nested.testzip():
                raise ValueError("Nested ZIP CRC failed: " + name)
    # Validate the unpacked code kit as well, independently of the code ZIP.
    kit = member(roots["code"], "final_report/" + VERSION + "/reproducibility")
    kit_count = 0
    for line in (kit / "files.sha256").read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        expected, name = line.split("  ", 1)
        if digest(member(kit, name).read_bytes()) != expected:
            raise ValueError("Unpacked code fingerprint failed: " + name)
        kit_count += 1
    if args.output:
        out = args.output.resolve()
        if out.exists() and any(out.iterdir()):
            raise FileExistsError("Choose a new or empty restore directory")
        for name, data in payload.items():
            path = member(out, manifest["archive_root"] + "/" + name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    if args.zip_output:
        target = args.zip_output.resolve()
        if target.exists():
            raise FileExistsError("Preserve the earlier ZIP: " + str(target))
        target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for entry in manifest["files"]:
                meta = entry["zip_metadata"]
                info = zipfile.ZipInfo(manifest["archive_root"] + "/" + entry["original_path"], tuple(meta["date_time"]))
                info.compress_type = meta["compress_type"]
                info.create_system = meta["create_system"]
                info.external_attr = meta["external_attr"]
                archive.writestr(info, payload[entry["original_path"]], compresslevel=9)
        if digest(target.read_bytes()) != manifest["source_archive_sha256"]:
            raise ValueError("Restored ZIP differs from the original archive")
    print(json.dumps({"passed": True, "delivery_files": len(payload), "original_fingerprints": count,
                      "nested_zip_crc_checks": 3, "code_fingerprints": kit_count,
                      "byte_identical_zip": bool(args.zip_output), "new_simulations": False}))

if __name__ == "__main__":
    main()

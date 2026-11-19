"""Stage the frozen TESS development cohort for a Google Drive folder upload."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".transiteye-workspace"
MANIFEST = WORK / "models/labeled_candidate_sequence_v1/cohort_manifest.json"


def stage() -> Path:
    raw = MANIFEST.read_bytes()
    original_sha = hashlib.sha256(raw).hexdigest()
    cohort = json.loads(raw)
    if cohort["protocol_id"] != "labeled_catalog_event_candidate_sequence_binary_v1":
        raise ValueError("Unexpected cohort protocol")
    out = WORK / "colab" / "TransitEye-FITS"
    out.mkdir(parents=True, exist_ok=True)
    products = []
    splits: dict[str, str] = {}
    total = 0
    for row in cohort["products"]:
        split = row["split"]
        group = row["source_group_id"]
        if splits.setdefault(group, split) != split:
            raise ValueError(f"Mixed partitions for {group}")
        if split == "fresh_holdout":
            continue
        if split not in {"train", "calibration", "selection"}:
            raise ValueError(f"Unexpected partition {split}")
        source = Path(row["relative_path"])
        sha = row["expected_sha256"]
        if len(sha) != 64 or any(ch not in "0123456789abcdef" for ch in sha):
            raise ValueError("Invalid product digest")
        relative = Path("fits") / sha[:2] / (sha + ".fits")
        dest = out / relative
        dest.parent.mkdir(exist_ok=True, parents=True)
        if dest.exists():
            if not dest.samefile(source) and (
                hashlib.sha256(source.read_bytes()).hexdigest() != sha
                or hashlib.sha256(dest.read_bytes()).hexdigest() != sha
            ):
                raise ValueError(f"Unexpected existing file {dest}")
        else:
            os.link(source, dest)
        total += source.stat().st_size
        products.append(dict(row, relative_path=relative.as_posix()))
    shutil.copytree(ROOT / "src/transiteye", out / "src/transiteye", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("uv.lock", "scripts/train_labeled_candidate_sequence.py",
                 "configs/labeled_candidate_sequence.v1.toml"):
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, out / name)
    cohort.update(products=products, original_manifest_sha256=original_sha)
    (out / "development.json").write_text(json.dumps(cohort, sort_keys=True))
    print(f"{len(products)} development product records, {total / 2**30:.2f} GiB. Upload {out}")
    return out


if __name__ == "__main__":
    stage()

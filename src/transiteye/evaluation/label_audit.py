"""Exploratory audit of the already-open historical v1 cohort's TRAIN labels only."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from transiteye.provenance import code_fingerprint

LABELS_SHA256 = "056ec18f2537383f35c775fbdda37408f3aedb48a2ca7261df01e42181388015"
SPLITS_SHA256 = "9fe940557d47ec98d7ed6d8e0c27a6b15733164728d106420b31da3e47bf4a79"


def summarize_training(labels: list[dict[str, Any]], assignments: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize TRAIN records only; never use label claims as model features."""
    splits: dict[str, str] = {}
    for item in assignments:
        group = item["source_group_id"]
        if group in splits:
            raise ValueError(f"duplicate source group in split manifest: {group}")
        splits[group] = item["split"]
    if len({item["source_group_id"] for item in labels}) != len(labels):
        raise ValueError("duplicate source group in label manifest")
    if any(item["source_group_id"] not in splits for item in labels):
        raise ValueError("label has no source-group partition")
    rows = [item for item in labels if splits[item["source_group_id"]] == "train"]
    return {
        "training_labels": len(rows), "class_counts": dict(Counter(item["label"] for item in rows)),
        "source_review_counts": {"|".join(key): count for key, count in
                                 Counter((item["label_source"], item["review_status"]) for item in rows).items()},
        "deep_false_positive_eclipse_proxies": sum(item["label"] == "eclipse_like"
            and "deep false positive" in item.get("notes", "").lower() for item in rows),
        "atl_variability_proxies": sum(item["label"] == "stellar_variability"
            and item["evidence_identifier"].endswith("-ATL") for item in rows),
        "other_artifact_explicit_toi_false_alarm_notes": sum(item["label"] == "other_artifact"
            and item.get("notes", "").startswith("TOI False Alarm instrumental trigger") for item in rows),
        "other_artifact_low_amplitude_notes": sum(item["label"] == "other_artifact"
            and "low-amplitude" in item.get("notes", "").lower() for item in rows),
        "drn_placeholder_identifiers": sum(item["evidence_identifier"].startswith("SPOC-DRN-") for item in rows),
    }


def run(labels_path: Path, splits_path: Path, output_path: Path) -> dict[str, Any]:
    """Audit only the checksummed, previously opened v1 manifests, never the EB holdout."""
    if (labels_path.resolve() != Path("manifests/real_catalog_cohort.v1.labels.json").resolve()
            or splits_path.resolve() != Path("manifests/real_catalog_cohort.v1.splits.json").resolve()):
        raise ValueError("audit only accepts the historical v1 manifest paths")
    labels_raw, splits_raw = labels_path.read_bytes(), splits_path.read_bytes()
    labels_hash = hashlib.sha256(labels_raw).hexdigest()
    splits_hash = hashlib.sha256(splits_raw).hexdigest()
    if labels_hash != LABELS_SHA256 or splits_hash != SPLITS_SHA256:
        raise ValueError("audit is restricted to the checksummed historical v1 manifests")
    result = {"schema_version": "1", "scope": "historical v1 TRAIN only; exploratory, no EB holdout",
              "labels_sha256": labels_hash, "splits_sha256": splits_hash,
              "backend_sha256": code_fingerprint(),
              **summarize_training(json.loads(labels_raw)["labels"], json.loads(splits_raw)["assignments"])}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path)
    parser.add_argument("splits", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.labels, args.splits, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

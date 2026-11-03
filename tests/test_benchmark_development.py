"""Preflight invariants for the frozen development workload."""

import hashlib
import json
from pathlib import Path

import pytest

from transiteye.execution import benchmark_development
from transiteye.io.manifest import ManifestEntry, ProductManifest, write_manifest


def _fixture(tmp_path: Path, *, invalid_checksum: bool = False, split: str = "train") -> tuple[Path, Path]:
    data = tmp_path / "data"
    data.mkdir()
    source = data / "target.fits"
    source.write_bytes(b"placeholder fits content")
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"assignments": [{"source_group_id": "tic-2", "split": split}]}))
    products = tmp_path / "products.json"
    products.write_text("{}")
    config = tmp_path / "benchmark.toml"
    config.write_text(f'''schema_version = "1"
split_manifest = "{splits}"
product_manifest = "{products}"
data_directory = "{data}"
target_count = 1
workers = 1
device = "cpu"
target_timeout_seconds = 30.0
pause_after_finished = 0
rss_sample_seconds = 0.25
[analysis]
max_candidates = 1
snr_threshold = 6.0
screen_snr_threshold = 5.0
bootstrap_replicates = 10
''')
    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    entry = ManifestEntry(product_id="id-2", data_uri="mast:example", target_id="2", sector=1,
                          product_type="tess_light_curve", local_filename=source.name,
                          expected_sha256="0" * 64 if invalid_checksum else sha(source))
    manifest = ProductManifest.create("test", (entry,), {
        "benchmark_config_sha256": sha(config), "split_manifest_sha256": sha(splits),
        "product_manifest_sha256": sha(products),
    })
    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest, manifest_path)
    return config, manifest_path


def test_preflight_verifies_checksum_and_records_effective_settings(tmp_path: Path, monkeypatch) -> None:
    config, manifest = _fixture(tmp_path)
    captured = {}

    def fake_benchmark(targets, output_dir, **kwargs):
        captured.update(targets=targets, output_dir=output_dir, **kwargs)
        return "benchmark-result"

    monkeypatch.setattr(benchmark_development, "run_sector_benchmark", fake_benchmark)
    assert benchmark_development.run(config, manifest, tmp_path / "output") == "benchmark-result"
    audit = json.loads((tmp_path / "output" / "preflight.json").read_text())
    assert audit["failures"] == 0
    assert captured["job_config"]["expected_checksums"]["id-2"] == audit["inputs"][0]["actual_sha256"]
    assert captured["job_config"]["exposure_status"] == "fixed_assumption_unverified"
    assert captured["job_config"]["exposure_seconds"] is None


def test_preflight_reports_checksum_failure_and_rejects_test_partition(tmp_path: Path) -> None:
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    config, manifest = _fixture(bad_dir, invalid_checksum=True)
    with pytest.raises(ValueError, match="preflight failed"):
        benchmark_development.run(config, manifest, bad_dir / "output")
    audit = json.loads((bad_dir / "output" / "preflight.json").read_text())
    assert audit["failures"] == 1
    assert "checksum mismatch" in audit["inputs"][0]["error"]
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    config, manifest = _fixture(test_dir, split="test")
    with pytest.raises(ValueError, match="non-TRAIN"):
        benchmark_development.run(config, manifest, test_dir / "output")

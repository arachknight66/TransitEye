from __future__ import annotations

from pathlib import Path

from transiteye.__main__ import main
from transiteye.config import ensure_workspace, load_config
from transiteye.datasets.labels import LabelRecord, LabelSource, MorphologyLabel, write_labels
from transiteye.datasets.splits import (
    DatasetSplit,
    SplitAssignment,
    SplitManifest,
    write_split_manifest,
)


def test_workspace_config_creates_external_directories(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(f'[workspace]\nroot = "{tmp_path / "workspace"}"\n', encoding="utf-8")
    config = load_config(config_path)
    ensure_workspace(config)
    assert config.workspace.cache_directory.is_dir()
    assert config.workspace.results_directory.is_dir()


def test_cli_exposes_version(capsys: object) -> None:
    try:
        main(["--version"])
    except SystemExit as error:
        assert error.code == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert "0.1.0" in captured.out


def test_cli_validates_manifest(tmp_path: Path, capsys: object) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"manifest_id":"demo","created_at":"2026-09-14T00:00:00+00:00","source":"test",'
        '"entries":[],"query":{},"schema_version":"1"}',
        encoding="utf-8",
    )
    assert main(["inspect-manifest", str(manifest)]) == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert '"entry_count": 0' in captured.out


def test_cli_audits_a_valid_cohort(tmp_path: Path, capsys: object) -> None:
    labels_path = tmp_path / "labels.json"
    splits_path = tmp_path / "splits.json"
    groups = ("one", "two", "three", "four")
    labels = tuple(
        LabelRecord(
            source_group_id=group,
            target_id=group,
            label=MorphologyLabel.TRANSIT_LIKE,
            label_source=LabelSource.MANUAL_REVIEW,
            evidence_uri="https://example.invalid/evidence",
            evidence_identifier=group,
            review_status="reviewed",
            snapshot_date="2026-09-14",
            sectors=(1,),
        )
        for group in groups
    )
    write_labels(labels, labels_path)
    write_split_manifest(
        SplitManifest.create(
            tuple(
                SplitAssignment(group, split)
                for group, split in zip(
                    groups,
                    (
                        DatasetSplit.TRAIN,
                        DatasetSplit.CALIBRATION,
                        DatasetSplit.SELECTION,
                        DatasetSplit.TEST,
                    ),
                    strict=True,
                )
            )
        ),
        splits_path,
    )
    assert main(["audit-cohort", "--labels", str(labels_path), "--splits", str(splits_path)]) == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert '"train": 1' in captured.out


def test_cli_analyze_fits_command(tmp_path: Path, capsys: object) -> None:
    import numpy as np
    from astropy.io import fits

    fits_path = tmp_path / "target.fits"
    t = np.linspace(1.0, 15.0, 1000)
    f = np.ones_like(t)
    # Inject dip
    f[500:520] = 0.99
    cols = [
        fits.Column(name="TIME", format="D", array=t),
        fits.Column(name="PDCSAP_FLUX", format="D", array=f),
        fits.Column(name="PDCSAP_FLUX_ERR", format="D", array=np.full_like(t, 0.001)),
        fits.Column(name="QUALITY", format="K", array=np.zeros(len(t), dtype=np.int64)),
        fits.Column(name="CADENCENO", format="K", array=np.arange(len(t), dtype=np.int64)),
        fits.Column(name="MOM_CENTR1", format="D", array=np.zeros(len(t))),
        fits.Column(name="MOM_CENTR2", format="D", array=np.zeros(len(t))),
    ]
    table = fits.BinTableHDU.from_columns(cols)
    primary = fits.PrimaryHDU()
    primary.header["TICID"] = 777666
    fits.HDUList([primary, table]).writeto(fits_path)

    out_dir = tmp_path / "bundle"
    exit_code = main(["analyze-fits", str(fits_path), "--output-dir", str(out_dir), "--progress-jsonl"])
    assert exit_code == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert '"bundle_path":' in captured.out
    assert (out_dir / "result.json").is_file()
    assert (out_dir / "arrays.npz").is_file()
    assert (out_dir / "candidates.parquet").is_file()


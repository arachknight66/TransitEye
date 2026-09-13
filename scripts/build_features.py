"""Build and freeze B039 matrices for the accepted demo and scientific datasets."""

from pathlib import Path

from transiteye.features import build_feature_matrix, freeze_feature_matrix


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    for dataset_version in (
        "dataset-demo-a74b6aad7c21faf15b0d",
        "dataset-4b84e8acaa6f6b334c2f",
    ):
        built = build_feature_matrix(root, dataset_version)
        output = freeze_feature_matrix(built, root / "data/features")
        print(f"{dataset_version}: {len(built.features)} rows -> {output}")


if __name__ == "__main__":
    main()

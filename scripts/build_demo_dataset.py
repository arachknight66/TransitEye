"""Build the controlled D001--D008 demo dataset from frozen local artifacts."""

from transiteye.demo import build_demo_dataset

if __name__ == "__main__":
    result = build_demo_dataset(".")
    print(result)

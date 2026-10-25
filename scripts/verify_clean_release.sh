#!/usr/bin/env bash
set -euo pipefail

echo "=========================================================="
echo "TransitEye Clean-Room Release Verification Suite"
echo "=========================================================="

# 1. Dependency lock verification
echo "[1/6] Verifying dependency lock..."
uv sync --locked --all-extras

# 2. Code style and lint check
echo "[2/6] Running ruff lint and format check..."
uv run ruff check .

# 3. Headless test suite (including scientific backend, desktop UI, architecture boundary)
echo "[3/6] Running full test suite headlessly..."
QT_QPA_PLATFORM=offscreen uv run pytest

# 4. CLI entry points
echo "[4/6] Verifying CLI entrypoints..."
uv run python -m transiteye --help > /dev/null
uv run transiteye --help > /dev/null

# 5. Scientific report compliance check
echo "[5/6] Verifying scientific report page limit (<= 3 pages)..."
if command -v pdfinfo > /dev/null 2>&1; then
    PAGE_COUNT=$(pdfinfo docs/report/transiteye_report.pdf | awk '/Pages:/ {print $2}')
    echo "Report page count: ${PAGE_COUNT}"
    if [ "${PAGE_COUNT}" -gt 3 ]; then
        echo "ERROR: Scientific report exceeds 3 pages!"
        exit 1
    fi
else
    echo "Warning: pdfinfo not installed, skipping page count check."
fi

# 6. Git clean state check (no large datasets tracked in git)
echo "[6/6] Checking Git hygiene..."
TRACKED_LARGE=$(git ls-files "*.fits" "*.fits.gz" "*.db" "*.parquet" "*.npz" || true)
if [ -n "${TRACKED_LARGE}" ]; then
    echo "ERROR: Large or generated data files are tracked in Git:"
    echo "${TRACKED_LARGE}"
    exit 1
fi

echo "=========================================================="
echo "SUCCESS: TransitEye clean-room release verification passed!"
echo "=========================================================="

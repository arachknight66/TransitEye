# Clean environment check, v1

On 2026-11-01, a separate Python 3.11.16 virtual environment was created on the same Linux host
as the original run. This is an environment reproduction, not an independent machine or reviewer.
The wheel SHA-256 was `a51b78dca89db90c9f3cdb681d92233d5d16f7aff20f95664babe732946d8238`;
the `uv.lock` SHA-256 was `c54b5ce138381d20b942ec5e3f96773e7fe674452ca7bf600f70bd8ca6e27f56`.
The build, package inventory, test logs, and comparison are retained in
`.transiteye-workspace/remediation/r4-clean-environment-v1/`.

From the repository root, the wheel was built with `uv build`, dependencies installed with
`uv sync --locked --extra desktop --no-install-project` using `UV_PROJECT_ENVIRONMENT` pointing
at the new environment, and the wheel installed into that environment with `uv pip install
--no-deps`. The installed package passed CLI help and all 165 tests with `QT_QPA_PLATFORM=offscreen`.
Three separate installed-wheel desktop and architecture runs passed 22 tests each; a native X11
run of the close/reopen and session/export tests passed two tests. The user also confirmed the
full manual Linux analysis, session, export and batch workflow on the current checkout. Automated
native tests do not substitute for that manual check.

The frozen real-substrate injection pilot was then run using the installed wheel, the original
selection file and the settings in `configs/real_substrate_injection_pilot.v1.toml`, with
`OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS` and `MKL_NUM_THREADS` set to 1. The backend fingerprint
matched the original, all 20 trial JSON records and the aggregate summary matched exactly, and all
20 reproduced result bundles passed integrity checks. The audit is in
`reproduced-real-pilot/comparison.json` under the external workspace path above.

This result checks that particular pilot only. It does not validate its scientific sensitivity,
reproduce the entire release on a different machine, or complete the independent classifier,
search-significance, physical-model, injection-coverage and final-report gates.

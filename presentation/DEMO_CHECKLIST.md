# Presentation-day checklist

## Before the demonstration

- [ ] Run `uv sync --all-groups --locked` in advance.
- [ ] Run `uv run python scripts/reproduce_project.py --verify`.
- [ ] Run `uv run python scripts/reproduce_project.py --report`.
- [ ] Run `uv run python scripts/reproduce_project.py --demo-smoke`.
- [ ] Confirm `results/demo_smoke.json` says `status: passed`.
- [ ] Open architecture, robustness, recovery, shift, and readiness figures.
- [ ] Make terminal and image-viewer fonts legible from the back of the room.
- [ ] Keep `results/index.json` available for artifact questions.

## During the demonstration

- [ ] Use frozen local artifacts; do not run network acquisition.
- [ ] Do not run the expensive full replay live.
- [ ] State that the demo uses synthetic signals in real TESS substrates.
- [ ] Separate BLS recovery from conditional classifier correctness.
- [ ] Lead with grouped LOTO, then qualify the two-TIC locked test.
- [ ] Describe scientific candidates as unlabeled scores.
- [ ] End with demo complete and scientific readiness blocked.

## Fallback

If live execution fails:

- [ ] Show the frozen `results/demo_smoke.json` output.
- [ ] Present the generated figures and CSV tables.
- [ ] Show the last checksum summary from `--verify`.
- [ ] Explain that report outputs are deterministic replays of frozen inputs.
- [ ] Continue from `presentation/DEMO_RESULTS.md` without network access.

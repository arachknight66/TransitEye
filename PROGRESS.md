# PROGRESS.md

## Status key

- `[x]` Completed and verified.
- `[~]` In progress; implementation or verification remains.
- `[ ]` Backlog; not started.
- `[!]` Blocked; reason and required resolution must be recorded.

## Project snapshot

**Current phase:** Phase 1 — awaiting reviewed pilot cohort  
**Current implementation:** Tasks 1.1–1.4 complete; task 1.5 ready for curated labels  
**Last updated:** 2026-09-14  
**Next task:** Add independently reviewed real label records, freeze the split manifest, and run the
Phase 1 data audit

## Completed

- [x] Removed the previous TransitEye implementation, scientific pipeline, datasets, artifacts,
  frontend, reports, project-specific tests, configuration, and documentation.
  - Evidence: The working tree contains only repository/Python setup, the package marker, README, and
    these root context documents; Git history remains intact.
- [x] Reduced the Python package to a clean importable foundation.
  - Evidence: `uv sync --locked` and `uv run python -c "import transiteye"` succeeded during cleanup.
- [x] Approved the rebuild architecture and scientific boundaries.
  - Decisions: Linux/CPU first; archived SPOC/TESS-SPOC light curves; targeted pixel vetting; separate
    morphology and source assessment; backend independent of PyQt; open-source PyQt6 distribution.
- [x] Created the AI development contract and project context.
  - Evidence: `AGENTS.md` and `PROJECT.md` describe working rules, current state, target architecture,
    technology stack, features, outputs, and exclusions.
- [x] Converted the approved architecture into a five-phase execution roadmap.
  - Evidence: `PLAN.md` defines ordered tasks, acceptance criteria, verification, phase gates, final
    deliverables, and the recommended MVP boundary.
- [x] 1.1 Established packages, tooling, configuration, structured logging, and a Qt-free CLI.
  - Evidence: `uv sync --locked`, `ruff`, package tests, and `python -m transiteye --help` pass.
- [x] 1.2 Defined versioned Qt-free domain/result contracts with JSON-safe serialization and explicit
  no-signal, insufficient-data, failed-fit, uncertainty, and classification states.
  - Evidence: Contract invariant and serialization tests pass.
- [x] 1.3 Implemented SPOC/TESS-SPOC FITS validation and delivered-light-curve ingestion.
  - Evidence: FITS fixtures verify flux selection, time/cadence preservation, quality/centroid loading,
  provenance, and malformed-input errors.
- [x] 1.4 Implemented MAST discovery, immutable manifests, checksum validation, retry/resume behavior,
  and CLI operations for discovery, acquisition, and manifest inspection.
  - Evidence: Mocked acquisition tests pass; live MAST smoke discovery returned 27 products for TIC
  307210830 on 2026-09-14.

## In Progress

- [~] Phase 1 — Contracts, ingestion, and curated data foundation.
  - Owner: Codex
  - Started: 2026-09-14
  - Remaining gate: independently reviewed real label snapshot and corresponding locked split manifest.
  - Verification: label evidence audit, class/support report, and source-group overlap check.

## Backlog

### Phase 1 — Contracts, ingestion, and curated data foundation

- [x] 1.1 Establish packages, tooling, and configuration.
- [x] 1.2 Define domain and result contracts.
- [x] 1.3 Implement FITS ingestion and validation.
- [x] 1.4 Implement MAST discovery and immutable manifests.
- [~] 1.5 Create label policy, pilot cohort, and locked split manifest.
  - Label/split schemas, validation, provenance fields, and CLI audit are complete.
  - A real cohort is intentionally pending independently reviewed source labels; no unreviewed catalog
    rows or fabricated negative labels will be committed as scientific ground truth.

### Phase 2 — Preprocessing, detection, and characterization baseline

- [ ] 2.1 Build quality assessment and dual light-curve representations.
- [ ] 2.2 Implement duration-aware detrending.
- [ ] 2.3 Implement blind BLS and variability searches.
- [ ] 2.4 Refine candidates and estimate parameters.
- [ ] 2.5 Estimate SNR, significance, and uncertainty.

### Phase 3 — Features, multiclass classification, and source vetting

- [ ] 3.1 Implement the audited feature registry.
- [ ] 3.2 Create controlled synthetic augmentation and recovery datasets.
- [ ] 3.3 Train and select the morphology classifier.
- [ ] 3.4 Calibrate confidence and add abstention.
- [ ] 3.5 Implement independent source vetting.

### Phase 4 — Sector execution and PyQt desktop product

- [ ] 4.1 Build persistent sector execution.
- [ ] 4.2 Freeze and run a sector benchmark.
- [ ] 4.3 Establish the separate desktop package and process bridge.
- [ ] 4.4 Implement the interactive scientific workflow.
- [ ] 4.5 Complete batch controls, exports, and visual polish.

### Phase 5 — Locked evaluation, scientific report, and release

- [ ] 5.1 Freeze the final evaluation protocol.
- [ ] 5.2 Run locked and robustness evaluations once.
- [ ] 5.3 Produce publication and user deliverables.
- [ ] 5.4 Perform clean-room release acceptance.

## Blockers

- [!] A scientifically usable pilot cohort and locked split require independently reviewed label records
  with evidence URIs. Catalog lookup alone cannot supply the required review judgment, especially for
  other/artifact and ambiguous sources.

## Progress log

| Date | Change | Verification | Result |
|---|---|---|---|
| 2026-09-14 | Scrapped the prior implementation and restored a minimal foundation | `uv sync --locked`; package import | Passed |
| 2026-09-14 | Added AI context and five-phase rebuild roadmap | Document consistency and repository checks | Passed |
| 2026-09-14 | Implemented Phase 1 software foundation | `uv sync --locked`; Ruff; 19 pytest tests; CLI help; live MAST discovery | Passed except reviewed cohort gate |

## Update protocol

When work begins, move exactly one primary task into In Progress and add its owner or agent, start
date, and intended verification. When it passes, move it to Completed with concrete evidence and add
a dated log row. Record failed checks and scientific limitations; do not erase them from history.

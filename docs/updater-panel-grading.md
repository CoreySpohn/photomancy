# Grading the updater panel

`tools/updater_verdict.py` turns the records of a finished panel run into
`docs/updater-verdict.csv`, applying the criteria of the
[acceptance protocol](updater-protocol.md) unchanged. The commands below are
run through `tools/updater_panel.py`.

## Commands

- **Progress.** `python tools/updater_panel.py --status RUN_DIR --stage panel`
  counts the done and planned units of each part of the panel stage (the
  default stage is `development`).
- **Audit.** `python tools/updater_panel.py --check-panel RUN_DIR` plans every
  panel unit and reports the missing, duplicate and unknown unit ids, records
  under another run hash, records at other budgets, and the provenance of the
  manifests. The exit code is 0 for a complete panel and 2 otherwise.
- **Grade.** `python tools/updater_panel.py --grade RUN_DIR
  --mechanics-evidence ATTEST.json --freeze-evidence FREEZE.json
  [--verdict-out docs/updater-verdict.csv] [--memory-mb MB] [--latency-ms MS]
  [--allow-dirty]` audits the directory and writes the table. The exit code is
  0 for a pass, 1 for a fail and 2 for an incomplete or refused panel. The
  overall verdict is derived (fail if any group fails, otherwise incomplete if
  any is, otherwise pass) and printed; it is not a table column.
- **Freeze report binding.** `--freeze` writes a `binding` block (run hash,
  budgets, alias tolerance) into its report. `--freeze-evidence` requires a
  tracked regular file in this photomancy repository, with both its index
  entry and its working bytes unmodified relative to `HEAD`. The grading
  report records the resolved `commit`, `blob_hash` and repository-relative
  `path` under `freeze_evidence`. Missing, untracked, edited, staged, outside
  or ambiguous JSON evidence cannot support an alias pass.

This integrity requirement is an implementation choice. Integrity rests on
**the committed file plus review**: Git binds the exact bytes to a commit,
and the report's binding identifies the run implementation, budgets and alias
tolerance. This detects edits to the supplied file after that commit, including
whitespace edits. It does not prove that the evidence was computed honestly,
recompute development or exact-check records, authenticate the reviewer, or
protect against fabricated evidence that is committed and accepted in review.
There is no signature or independent attestation of the freeze computation.

`RUN_DIR` holds the `units-{i}of{n}.jsonl` and `manifest-{i}of{n}.json` files
of every shard.

## What the audit refuses

The audit refuses to grade, and writes no table, when any of these holds:

- a duplicate unit id, a record that is not a panel unit, or a record without
  an id;
- a record whose run hash is not the current one (another code or protocol
  content), or whose budget is not the protocol's;
- a manifest that is unreadable, missing for a unit file, recorded under
  another run or protocol hash, or from a run at test budgets;
- manifests that disagree on the JAX or jaxlib version, the host architecture
  (`arm64` and `aarch64` are one), the `--chunk` value, the code commit or the
  protocol commit;
- an invocation whose manifest lacks a populated provenance field (protocol and
  code commits as 40-digit hex, a boolean dirty flag, JAX and jaxlib versions,
  platform, argv), or whose recorded budgets dictionary differs from the
  protocol's;
- an invocation whose code tree was dirty. `--allow-dirty` accepts it for an
  exploratory grade only: the code commit column carries `-dirty`, no row may
  be `pass`, the panel is at best `incomplete` with the reason stated, and no
  table is written unless `--verdict-out` is given explicitly;
- a protocol whose status is not `frozen`.

Missing units do not refuse. The criteria that read a missing unit are
`incomplete` for its group, every other group is graded, and the panel verdict
follows the protocol: `fail` if any group fails, otherwise `incomplete` if any
is, otherwise `pass`. An incomplete panel is never a pass.

## What each criterion reads

Protocol citations below refer to `docs/updater-protocol.md`. Choices of
serialization, input validation and evidence binding are implementation rules,
not additional claims made by the protocol.

- **Coverage and calibration** apply the cell criteria in lines 893-905 and
  909-923. Selecting the nominal `updater_a` for these criteria and failed
  updates is an implementation ruling fixed before results. These rows never
  pool `updater_a` with `updater_b` or `updater_doubled`.
- **Alias accuracy.** D2 pairs `updater_a` with `reference_1` and `updater_b`
  with `reference_2`, applying the four-independent-run formula in lines
  931-940 and the bound and power requirements in lines 945-955. A reference
  that is uncertified or fails its exact checks makes only the affected mass model
  incomplete, and inadequate power makes only its group and quantity incomplete
  (lines 640-652 and 1216-1220). Such rows are `incomplete` even if their
  numerical D2 bound fails. Other certified models and powered groups retain
  their own verdicts, even when the freeze report is globally `incomplete`.
  A tolerance above `delta2_max` leaves every alias verdict `incomplete`.
  The binding and schema checks are implementation rules: require the three
  canonical exact cases (`ti_window`, `physical_fixed`, `ti_binding`), one
  power row per canonical group and quantity, and finite power probabilities
  in [0, 1]. TI's binding-window check belongs to TI. Missing, duplicate or
  contradictory group rows cannot certify their group. Unbound or malformed
  evidence blocks passes, while an independently computed failure remains
  `fail` unless an identified scoped deficiency makes that row incomplete.
- **Failed updates** count every declared non-OK status (`NO_SUPPORT`,
  `BUDGET_EXHAUSTED`, `LOW_ESS`, `INCOMPLETE_LADDER`) of `updater_a` per update,
  written in `exhausted_count` (lines 966-992). The cutoff and reported bound
  follow lines 994-999. As an input-validation choice, a missing or unknown
  status makes the group `incomplete`, never a count.
- **Schedule stability** compares `updater_a` with `updater_b` and with
  `updater_doubled`, using the comparisons, tolerance and count in lines
  1008-1012. Input validation makes a missing or out-of-range schedule
  probability `incomplete`.
- **Every world counts in every denominator.** A returned belief is scored
  whatever its status (lines 834-835), and a group with missing worlds is
  `incomplete` (lines 1214-1215). The implementation treats evidence that
  cannot be evaluated as `incomplete`, never a miss: invalid bin masses,
  missing or non-finite/nonpositive Kish ESS, a retained-particle count that
  is not a positive integer, a non-finite or unordered interval, a non-finite
  truth, and a PIT or schedule probability outside [0, 1].
- **Mechanics** requires green mechanics tests at the graded code commit,
  and any red test fails (lines 1027-1031). The implementation reads an
  attestation JSON from `--mechanics-evidence`: `result` (`pass` or `fail`),
  `code_commit` equal to the manifest commit, and `log_sha256` of the test log.
  An optional `log_path` beside the file is re-hashed. A missing, unbound or
  incomplete attestation gives `incomplete`, and a bound recorded `fail`
  gives `fail`. This verdict is written on every row.
- **Memory and latency** are reported, not graded (lines 1207-1210, 1265).
  `--memory-mb` is the measured peak resident memory and `--latency-ms` the
  measured warm batched cost per update. As a serialization choice, each is
  written on the failed-update rows only as given, with an empty cell when
  absent. Nothing is derived from per-world times.
- **Provenance.** Refusing duplicates and disagreeing manifests, requiring
  committed freeze evidence, and marking dirty overrides exploratory are
  implementation safeguards. Exploratory counts are calculated from the
  final rows after passes have become incomplete. Overall aggregation follows
  lines 1225-1227: any graded failure fails the panel, otherwise any incomplete
  group makes it incomplete. A globally incomplete freeze cannot yield a panel
  pass, even when every computable row passes.

`seeds` is the root seed and the stage. The protocol and code commits come
from the manifests.

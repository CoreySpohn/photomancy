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
  budgets, alias tolerance) into its report; `--freeze-evidence` reads that
  committed report.

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

Each ruling names the protocol line it follows (`docs/updater-protocol.md`).

- **Coverage and calibration** read the nominal-budget run `updater_a` of each
  world of a cell and never pool it with `updater_b` or `updater_doubled`
  (line 40: nominal-budget runs; lines 825-826: 400 worlds per cell).
- **Alias accuracy.** D2 pairs `updater_a` with `reference_1` and `updater_b`
  with `reference_2`, so each world needs four independent runs (line 592: the
  independent-run design that makes the cross product an unbiased squared
  error). It is `pass` only with freeze evidence bound to this run hash,
  budgets and tolerance, every mass model certified, passing exact checks and
  the power floor met for the group and quantity (lines 640-652: an uncertified
  reference makes alias `incomplete`). Without the evidence a computed failure
  stays `fail` and a pass becomes `incomplete`. A tolerance above
  `delta2_max` leaves every alias verdict `incomplete`.
- **Failed updates** count every declared non-OK status (`NO_SUPPORT`,
  `BUDGET_EXHAUSTED`, `LOW_ESS`, `INCOMPLETE_LADDER`) of the `updater_a` run per
  update, written in `exhausted_count` (lines 37-38 and 964). A missing or
  unknown status makes the group `incomplete`, never a count.
- **Schedule stability** compares `updater_a` with `updater_b` and with
  `updater_doubled`, with the protocol's tolerance and count. A missing or
  out-of-range schedule probability makes the group `incomplete`.
- **Every world counts in every denominator.** A returned belief is scored
  whatever its status (lines 834-835). Evidence that cannot be evaluated is
  `incomplete`, never a miss: bin masses that are not nonnegative and
  normalized, an empty reference fit (`particles` 0 or `kish_ess` 0), a
  non-finite or unordered interval, a non-finite truth, and a PIT or schedule
  probability outside [0, 1].
- **Mechanics** is an attestation JSON given to `--mechanics-evidence`:
  `result` (`pass` or `fail`), the `code_commit` the tests ran at (it must equal
  the commit in the manifests) and `log_sha256` of the test log (line 105 and
  the Mechanics section, line 1025). An optional `log_path` beside the file is
  re-hashed. A missing, unbound or incomplete attestation gives `incomplete`;
  a recorded `fail` gives `fail`. It is written on every row.
- **Memory and latency** are reported, not graded (lines 1207-1210, 1265).
  `--memory-mb` is the measured peak resident memory and `--latency-ms` the
  measured warm batched cost per update; each is written on the failed-update
  rows only as given and decides nothing. Without the flag the cell is empty;
  nothing is derived from per-world times.
- **Provenance.** Duplicates and disagreeing manifests are refused, and an
  override never restores an ordinary pass.

`seeds` is the root seed and the stage. The protocol and code commits come
from the manifests.

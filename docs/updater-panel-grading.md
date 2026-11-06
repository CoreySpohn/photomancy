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
- **Grade.** `python tools/updater_panel.py --grade RUN_DIR --mechanics pass
  [--verdict-out docs/updater-verdict.csv] [--memory-mb MB] [--allow-dirty]`
  audits the directory and writes the table. The exit code is 0 for a pass, 1
  for a fail and 2 for an incomplete or refused panel.

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
- an invocation whose code tree was dirty (`--allow-dirty` accepts it and marks
  the code commit column with `-dirty`);
- a protocol whose status is not `frozen`.

Missing units do not refuse. The criteria that read a missing unit are
`incomplete` for its group, every other group is graded, and the panel verdict
follows the protocol: `fail` if any group fails, otherwise `incomplete` if any
is, otherwise `pass`. An incomplete panel is never a pass.

## What each criterion reads

- **Coverage and calibration** read the nominal-budget run of the first key
  (`updater_a`) of each world of a cell.
- **Alias accuracy** pairs `updater_a` with `reference_1` and `updater_b` with
  `reference_2` in the D2 statistic. A reference fit with no weighted draw, or
  a non-finite bin mass, leaves its group `incomplete`. A frozen $\delta_2$
  above its stated maximum leaves every alias verdict `incomplete`.
- **Failed updates** count the worlds whose `updater_a` update status is not
  `OK`, per update; the count is written in `exhausted_count`.
- **Schedule stability** compares `updater_a` with `updater_b` and with
  `updater_doubled`. A schedule probability that cannot be evaluated counts as
  a deviation.
- **Mechanics** is the result of the mechanics tests at the graded code
  commit. It cannot be read from the records, so `--mechanics pass|fail` states
  it; without the flag the verdict is `incomplete`. It is written on every row.
- A cell whose world has a non-finite truth, interval or PIT counts that world
  as not covering, and the cell fails calibration.

`latency_ms_per_update` is the mean warm wall time of one update of one world
in the `updater_a` and `updater_b` runs, not the batched amortized cost, and
decides nothing. `memory_mb` is written on the failed-update rows from
`--memory-mb` and is empty without it. `seeds` is the root seed and the stage.
The protocol and code commits come from the manifests.

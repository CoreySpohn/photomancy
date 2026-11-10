# Updater development-panel record

Outputs of the first development run of the orbit-updater acceptance protocol
(`docs/updater-protocol.md`), at the starting budgets: `freeze.json` (freeze
status `incomplete`, with its reasons), `ess.json` (Kish ESS per regime, epoch
count and group) and `exact-checks.json` (refinement tables and verdicts of the
exact checks).

The records were produced with run hash `0ce279f74ab373c6bbdeb67ceada3461b30f5d9c953b69185ccaf02570a05ea1`
(exact-check run hash `58b8d22528f4ad70113a4b7718904c7c875446b0d31a2a4255959c449038ef19`) at protocol
and code commit `ec21b303a7211abe1ecc2ff8a60b81d734b72a7c`.

`budget-2p24/` holds the same three outputs for the rerun at the physical
reference budget of 2^24 proposals, at code commit
`564422b5627735caaca10b4340c7e60607b31840`. Its freeze status is `incomplete`, with the reasons in
`freeze.json` and the limit recorded in the protocol. The run hash is unchanged.

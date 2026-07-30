# interface_baselines (launchers only)

The shared Python implementation that used to live here (planner training,
command publication, low-level tracking, evaluation, audits, provenance) now
lives in the installable package `source/imitation_experiments/`, with its
tests under `source/imitation_experiments/tests/`. This directory keeps only
the frozen campaign shell launchers, which invoke the package with
`python -m imitation_experiments.<subpackage>.<module>`.

Do not add new Python modules here; add them to the package with a test.

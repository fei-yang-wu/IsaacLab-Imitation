# Paper-facing reproduction entrypoint

This directory is the stable public surface intended for the eventual
open-source release. It contains no diagnostic command styles, recovery jobs,
or historical launchers.

It holds only the guarded entrypoints — [`run.sh`](run.sh), the Phase-4 and
Phase-5 submit plus aggregate scripts, the release-bundle builder, and the
reference-buffer workflow. Shared implementation, diagnostics, and tests live in
the campaign that owns them, so this surface stays stable as campaigns come and
go. See [`../README.md`](../README.md) for the campaign layout.

## Script standard for this directory

Every script here must be **self-contained, current, and runnable as-is**: it
runs from the repository root with no undocumented prerequisite and no
hand-editing, every configurable parameter lives in the script as a named
constant or in a config file beside it (Hydra preferred, YAML under
[`conf/`](conf/)), it stays updated when the code it drives changes, and it
fails loudly rather than silently doing less work.
[`reference_buffer_workflow.py`](reference_buffer_workflow.py) with
[`conf/reference_buffer.yaml`](conf/reference_buffer.yaml) is the reference
implementation of that shape.

## Command-interface planner-capacity study

[`run_interface_capacity_study.py`](run_interface_capacity_study.py) with
[`conf/interface_capacity.yaml`](conf/interface_capacity.yaml) reproduces the
interface ablation and the capacity grid end to end. It asks whether a
compressed command interface reduces planner-training complexity, across a
packet-size ladder at a fixed 5 Hz publication rate:

| Interface | Packet | Qualified oracle |
| --- | --- | --- |
| `full_body_trajectory` | 670 | 23.8 mm |
| `root_qpos` | 380 | 23.6 mm |
| `ee_trajectory` (rootless control, off by default) | 360 | 405.2 mm |
| `latent_skill` | 258 | 30.5 mm |
| `root_points5` | 240 | 30.6 mm |

Each interface has its **own** natively-trained low-level controller, so no
channel is ever reconstructed. Planner rows are normalized by their own
interface's oracle, because the trackers differ in quality and absolute MPJPE
alone would conflate planner quality with tracker quality.

```bash
# everything: qualify -> oracle -> grid -> aggregate
pixi run python experiments/paper/run_interface_capacity_study.py

# one cell, for a smoke check
pixi run python experiments/paper/run_interface_capacity_study.py \
    grid.interfaces=[root_points5] grid.seeds=[0] grid.sizes=[tiny]
```

`qualify` gates everything downstream: an interface whose own controller cannot
track it makes every planner number on it meaningless. That is not hypothetical —
`ee_trajectory` reads a healthy 41.3 mm on the *training* metric while its true
frame-0/700-step floor is 405.2 mm, because training episodes terminate before
drift accumulates. Only the gate separates those cases, so never qualify an
interface on training-log MPJPE.

The study fails loudly rather than degrading: it verifies every checkpoint
against a recorded sha256, refuses an interface with no `PASS` qualification
record, refuses a non-empty study root, and writes `study_provenance.json`
(resolved config, resolved checkpoints, git commit) beside the artifacts.

Implementation lives in
`experiments/campaigns/2026-07-23-lafan1-planner-capacity/`; this entrypoint
supplies its inputs and enforces the gates rather than duplicating the
orchestration.

Current state: **controllers qualified, grid in progress**.

## Enc380 planner-route comparison

[`run_enc380_planner_route_comparison.py`](run_enc380_planner_route_comparison.py)
with [`conf/enc380_planner_route.yaml`](conf/enc380_planner_route.yaml) compares
two decomposed routes into one frozen tracker: `root_qpos planner -> frozen
encoder -> latent tracker` and `latent planner -> latent tracker`. It executes the
guarded sequence `qualify -> one paired oracle collection -> 12
capacity/seed cells -> aggregate`. One persistent 100-environment Isaac session
asynchronously collects 100 complete variable-length `walk1_subject1` segments;
partial live segments at the cutoff are discarded. Both routes are trained once
from the same paired causal rows. Planner sizes use 10k/20k/30k/50k updates,
respectively, at effective batch 1024, with memory-adjusted microbatches and
held-out best-checkpoint selection. ICE maps the 12 logical cells to 24
single-route jobs so each long fit has a full 16-hour walltime. There is no
learned-planner rollout collection or finetune stage. This user-selected
continuity motion is a preliminary diagnostic, not a representative paper
motion sample.

```bash
# Resolve config, verify the intended commands, and touch nothing.
pixi run python experiments/paper/run_enc380_planner_route_comparison.py

# Run only after replacing/confirming all paths and hashes in the Hydra config.
pixi run python experiments/paper/run_enc380_planner_route_comparison.py execution.dry_run=false
```

The checked-in config pins the completed 5B tracker and encoder hashes. A real
run remains blocked until the corrected 500-control-step, `[0, 200]`-start
qualification passes.

## Reference buffer workflow

Training reads the reference motion set through a TorchRL replay buffer.
Reconstructing it per job is expensive at scale — for the 129,785-clip
BONES-SEED set, about 4.3 h to build the Zarr from NPZ and a further 3.1 h to
fill the buffer. [`reference_buffer_workflow.py`](reference_buffer_workflow.py)
makes that a one-time cost and turns the buffer into a published artifact:

```bash
# once, where the NPZ tree lives
pixi run python experiments/paper/reference_buffer_workflow.py stages='[build,pack,push]'

# on a compute node, before training
pixi run python experiments/paper/reference_buffer_workflow.py \
    stages='[fetch]' paths.fetch_dest=/tmp/rb
```

Measured: 26.27 s to build versus **0.16 s to reopen**, byte-identical, with the
Zarr absent — so the Zarr is consumed exactly once and never needs to reach a
compute node. The published form is CPU memmap files; at train time
`env.dataset_storage_device=cuda:0` materializes it into VRAM in one sequential
read, leaving training throughput unchanged. Point training at it with:

```text
env.dataset_storage_device=cuda:0
env.dataset_storage_persist_dir=/tmp/rb
env.dataset_storage_persist_id=<buffer.persist_id from the config>
```

When the buffer omits the `next_*` keys (the default key subset, which is what
makes the full 129k set fit in an H200 alongside 4096 environments), training
must additionally set `env.reconstructed_reference_action=false` and
`env.observations.policy_supervision=null`.

Current state: **staging**.

- Phase 4 remains blocked on its corrected-LAFAN1 low-level gate.
- The first Phase-5 three-seed preparation failed; its default data budget and
  storage workflow still need the revisions recorded in
  [`wiki/current-status.md`](../../wiki/current-status.md).
- The final release bundle requires complete audited aggregates from both
  phases.

Accordingly, [`run.sh`](run.sh) reports the intended commands but refuses to
launch them while `PAPER_WORKFLOW_STATE=staging`. Internal development
continues through the canonical guarded scripts in
[`experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/`](../campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/).

The eventual public commands are:

```bash
experiments/paper/run.sh phase4-lafan1
experiments/paper/run.sh phase5-bones-seed
experiments/paper/run.sh bundle \
  --phase4_aggregate /path/to/phase4/aggregate \
  --phase5_aggregate /path/to/phase5/aggregate \
  --output_dir /path/to/new/release
```

Before changing the state to `ready`, require:

1. the current paper protocol and public commands agree;
2. both phase launchers pass their dry runs and fixed gates;
3. both audited aggregates exist;
4. the bundle builder passes without weakened checks;
5. README paths and a clean-checkout reproduction are verified.

The authoritative scope is exactly two planner rows: DiffSR latent commands
and the ten-frame explicit vanilla packet. The direct vanilla tracker is a
low-level ceiling, not a third planner row.

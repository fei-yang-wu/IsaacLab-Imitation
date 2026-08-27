# 2026-08-21 — Interface combinations at the 2B screen

Do the winning spokes of the 2026-08-19 interface design study stack? The star
measured main effects only. This campaign trains the five combination and
follow-up cells its results point at, on the byte-identical protocol (2B
matched frames, the same hub configuration, data, recipe, and milestone
cadence), so every row reads directly against the star's table.

W&B group: `interface-combos-2b` (confirmed with the user 2026-08-21).
Seed 0 only at the screen, like the star: a single-arm difference inside the
~15% band is directional, not resolved.

## Reference rows (the star, 2B, seed 0, `paper_testbed4096_v1`)

| arm | SR | MPJPE-L | MPJPE-G |
|---|---:|---:|---:|
| `ctrl` (hub) | 0.9023 | 24.49 mm | 212.3 mm |
| `use_hold1` | 0.8921 | 26.44 mm | 150.4 mm |
| `obj_jepa_sigreg_ebm` | 0.8875 | 27.70 mm | 212.8 mm |
| `obj_recon` | 0.8931 | 27.50 mm | 446.3 mm |
| `bn_sonic_fsq64` | 0.8701 | 29.07 mm | — |
| `ix_fsq64_hold1` | 0.8496 | 30.45 mm | 136.9 mm |
| `use_phase_none` (hold 10) | 0.3679 | 66.48 mm | — |

## Arms and what each one tests

| arm | changes vs `ctrl` | parents | hypothesis |
|---|---|---|---|
| `jepa_ebm_hold1_256d` | objective `jepa_ntp/sigreg_ebm` + hold 1 | `obj_jepa_sigreg_ebm`, `use_hold1` | the best objective and the best-global interface stack |
| `jepa_ebm_hold1_fsq64` | + SONIC FSQ 64x32 token space | `ix_fsq64_hold1`, `bn_sonic_fsq64` | the same pair survives the planner-facing discrete interface |
| `recon_endpoint` | recon target `endpoint` | `obj_recon` | the +110% MPJPE-G of `obj_recon` is the input-window target, not reconstruction itself |
| `recon_full_window` | recon target `full_window` | `obj_recon` | same question, decoding all ten future slots incl. the hidden endpoint |
| `hold1_live_phase` | hold 1 + `command_phase_source=episode`, period 10 | `use_hold1`, `use_phase_none` | a live clock returns to hold-1 arms the channel that is load-bearing at hold 10 |

`sigreg_ebm` already contains the endpoint DiffSR loss (it is endpoint + chunk
NTP + SIGReg), so `jepa_ebm_hold1_256d` is also the "endpoint plus JEPA
auxiliary" cell; no separate arm is needed for that.

The two `jepa_ebm_hold1_*` arms change TWO fields against `ctrl` on purpose:
they are combination cells and their single-change parents are in the star.
State both parents when reading a row.

New code these arms depend on (2026-08-21, RLOpt):

- `HighLevelSkillDiffSRConfig.reconstruction_target`
  (`input_window` default | `endpoint` | `full_window`), CLI
  `--reconstruction_target`. Old checkpoints load as `input_window`.
- `ipmd.latent_learning.command_phase_source` (`hold` default | `episode`) and
  `command_phase_period` (0 = `code_period`). `episode` requires hold 1 and is
  wired into the `hl_skill` sampler only, not the skill-commander path.
- Tests: `RLOpt/tests/test_hl_skill_recon_phase.py` (registered in
  `pixi.toml` `test-rlopt`).

## Pipeline

```bash
./smoke.sh                      # local wiring qualification, all arms
./plan_all.sh                   # resolve every plan offline
./submit.sh <arm> 0             # plan one arm; prints the submit --confirm line
./mirror.sh                     # pull checkpoints off ICE
./eval.sh                       # milestone + clean + robust rows
./eval.sh --report
```

`submit.sh` only plans. Nothing reaches Slurm until the printed
`submit --plan <dir> --confirm <PLAN_SHA>` command is run.

Evaluation reads every per-arm interface field (width, hold, phase mode,
phase source, phase period, macro terms, stride, anchor) back out of
`campaign.yaml`, so it cannot drift from training.

## Follow-up already decided

The 20B SONIC-reset continuation (`2026-08-18-sonic-reset-20b`) lifted both
10B leaders (`ln_hold1_sonicreset` 0.9558 SR / 22.15 mm at 20B, one seed,
resets and frames confounded by design). Any arm promoted from this screen to
10B carries the SONIC reset recipe in its continuation, not at the screen —
the screen must stay byte-comparable to the star.

## Results (2026-08-21, all five arms at exactly 2,000,289,792 frames)

Clean board (`bones_testbed4096_v1`), one seed, run-to-run SR floor ~0.001-0.003;
robust (`no_push`) rows and the eight-point milestone curves are in
`logs/interface_combos_eval/`.

| arm | SR | MPJPE-L | MPJPE-G | ee_body_pos |
|---|---:|---:|---:|---:|
| star `ctrl` (reference) | 0.9023 | 24.49 mm | 212.3 mm | — |
| `recon_endpoint` | 0.8992 | **24.16 mm** | 373.1 mm | 319 |
| `recon_full_window` | 0.8992 | 25.79 mm | 453.0 mm | 365 |
| `hold1_live_phase` | 0.8945 | 26.33 mm | 150.7 mm | 345 |
| `jepa_ebm_hold1_256d` | 0.8918 | 27.50 mm | **142.0 mm** | 387 |
| `jepa_ebm_hold1_fsq64` | 0.8499 | 31.64 mm | **129.6 mm** | 544 |

Read against the parents (one seed, directional):

- **No combination beats `ctrl` on SR.** The star's conclusion reproduces;
  no arm here earns a 10B promotion on success-rate grounds.
- **`recon_endpoint` vs `obj_recon`: local 27.50 -> 24.16 mm (-12%), global
  446 -> 373 mm (-16%), SR +0.006.** The endpoint decode target repairs part
  of the drift the input-window target caused — but only part, and
  `recon_full_window` repairs none of it (453 mm). The refined claim: the
  input-window TARGET was the worst choice, but reconstruction as an
  objective still drifts relative to endpoint DiffSR.
- **`hold1_live_phase` is a NULL result** — every axis within noise of
  `use_hold1` (0.8921 / 26.44 / 150.4). At hold 1 the re-encoded command
  already carries the timing; the phase channel matters only at long holds.
- **JEPA x hold-1 stacks on GLOBAL error, not SR**: 142.0 mm (parents 150.4
  and 212.8) and 129.6 mm for the fsq64 version — the two best MPJPE-G
  values in the program. The fsq64 arm's SR matches `ix_fsq64_hold1`
  exactly (0.8499 vs 0.8496): the objective change does not move SR in the
  discrete hold-1 cell.

## Status

- 2026-08-21: training complete on all five arms (~4.4 h each, segment 1
  only); 50 eval rows scored locally, zero failures.
- 2026-08-21: campaign created; every arm passed the local wiring smoke
  (pretrain 4 updates at production batch 8192, one 128-frame IPMD iteration,
  `smoke_verdict` pass on all five; the fsq arm's code perplexity 27.0 at 4
  updates). All five arms SUBMITTED to ICE, seed 0: `jepa_ebm_hold1_256d`
  5587411-13, `jepa_ebm_hold1_fsq64` 5587414-16, `recon_endpoint` 5587417-19,
  `recon_full_window` 5587420-22, `hold1_live_phase` 5587423-25
  (pretrain -> lowlevel1 -> lowlevel2 per arm). Working tree carried the
  same-day RLOpt changes listed above; drift recorded by `submit`. Nothing
  measured.

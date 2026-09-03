# 2026-09-02 -- every live arm at its latest checkpoint

One `eval_checkpoint_tree.py --final_only` job per arm on the arm's real
trainer tree, star-v2 curves evaluator arguments, `bones_testbed4096_v1`
clean, `--randomization none`, seed 0. Rows land in
`/data/eval/latest_eval/<arm>_seed0_clean_f<frames>.json` on ICE.

| arm | campaign | checkpoint at submit |
|---|---|---|
| `z64_merged` | latent64-probe-10b | 9.50B (walltime end) |
| `enc_hist` | latent64-probe-10b | 9.50B (walltime end) |
| `obs_hist` | latent64-probe-10b | 8.50B (walltime end) |
| `z64_wd_clin` | latent64-probe-10b | 7.0B, still training |
| `lstm` | lstm-hub64-10b | 4.5B, still training |
| `lstm_affine` | lstm-hub64-10b | 4.5B, still training |

The LSTM rows carry the recurrent-state qualification debt from
`2026-08-28-smooth-ablation-5b` (see the campaign.yaml header).

Inspection clips: `./render_clips.sh` (workstation, PhysX, studio look) on
the same checkpoints mirrored to `logs/latent64_probe_mirror/ckpt/`.

## Rows (2026-09-02, jobs 5626313/14/15/17/19/20)

One seed, `bones_testbed4096_v1` clean, checkpoints as listed (frames differ
per row; the `lstm*` rows are 4.5B of 10B and still training; the
`z64_wd_clin` row is 7.0B of a run that ends near 9.6B). Three of the
nine job starts died in the flaky Kit startup crash and were resubmitted.

| arm | checkpoint | SR | MPJPE-L | MPJPE-G | acc | jerk | action_delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| `z64_merged` (control) | 9.50B | 0.9292 | 23.25 | 93.09 | 4.25 | 174.9 | 0.780 |
| `enc_hist` | 9.50B | 0.9304 | 22.57 | 101.66 | 4.39 | 185.8 | 0.802 |
| `obs_hist` | 8.50B | 0.9233 | 23.84 | 84.21 | 4.79 | 213.6 | 0.897 |
| `z64_wd_clin` | 7.00B | 0.9312 | 22.10 | 90.37 | 4.29 | 178.3 | 0.763 |
| `lstm` | 4.50B | 0.8884 | 23.98 | 136.49 | 4.98 | 212.7 | 0.851 |
| `lstm_affine` | 4.50B | 0.8840 | 25.51 | 145.46 | 5.11 | 220.3 | 0.895 |
| `sonic_v1_1` | released | 0.9888 | 26.73 | 187.7 | 3.45 | - | - |

## Final rows (2026-09-02 evening, jobs 5627493 / 5627487 / 5627490)

One seed, `bones_testbed4096_v1` clean. `z64_wd_clin` ended on its 15:59
walltime at 9.62B (last save 9.5B). The LSTM arms finished their 10B budget
in two segments (resume from 8.5B). Earlier rows of the same arms: 7.0B
`z64_wd_clin` 0.9312 / 22.10 / 90.37; 4.5B `lstm` 0.8884 / 23.98 / 136.49;
4.5B `lstm_affine` 0.8840 / 25.51 / 145.46.

| arm | checkpoint | SR | MPJPE-L | MPJPE-G | acc | jerk | action_delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| `z64_wd_clin` | 9.50B (walltime end) | 0.9395 | 21.74 | 83.84 | 4.28 | 180.1 | 0.771 |
| `lstm` | 10.0B (final) | 0.9121 | 21.24 | 103.11 | 4.76 | 207.0 | 0.857 |
| `lstm_affine` | 10.0B (final) | 0.9062 | 22.27 | 110.63 | 4.78 | 205.4 | 0.878 |
| `combo` | 10.0B (final) | 0.9214 | 22.64 | 88.52 | 4.53 | 195.5 | 0.840 |
| `sonic_v1_1` | released | 0.9888 | 26.73 | 187.7 | 3.45 | - | - |

## Final-checkpoint rows (2026-09-02 evening, jobs 5627493 / 5627487 / 5627490)

One seed, `bones_testbed4096_v1` clean. The LSTM trees carry two run
directories after the resume, so their evals score a milestone-layout
symlink tree at `/data/lstm_hub64_final/<arm>_seed0/tracker/f10000269312/`.
`z64_wd_clin`'s eval needed four submissions (three Kit startup crashes).

| arm | checkpoint | SR | MPJPE-L | MPJPE-G | acc | jerk | action_delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| `z64_wd_clin` | 9.50B (final) | 0.9395 | 21.74 | 83.84 | 4.28 | 180.1 | 0.771 |
| `lstm` | 10.0B (final) | 0.9121 | 21.24 | 103.11 | 4.76 | 207.0 | 0.857 |
| `lstm_affine` | 10.0B (final) | 0.9062 | 22.27 | 110.63 | 4.78 | 205.4 | 0.878 |
| `z64_merged` (control) | 9.50B | 0.9292 | 23.25 | 93.09 | 4.25 | 174.9 | 0.780 |
| `enc_hist` | 9.50B | 0.9304 | 22.57 | 101.66 | 4.39 | 185.8 | 0.802 |
| `sonic_v1_1` | released | 0.9888 | 26.73 | 187.7 | 3.45 | - | - |

| `combo` | 10.0B (final) | 0.9214 | 22.64 | 88.52 | 4.53 | 195.5 | 0.840 |

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

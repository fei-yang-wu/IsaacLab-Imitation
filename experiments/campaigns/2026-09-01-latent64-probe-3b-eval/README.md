# 2026-09-01 -- latent64-probe-10b arms scored at 3B

Scores `z64_merged`, `enc_hist`, and `obs_hist` from
`../2026-09-01-latent64-probe-10b` at the 3B milestone
(`model_step_3000238080.pt`, cumulative frames 3,000,238,080 in every arm)
on the star-v2 board: `bones_testbed4096_v1`, clean row, `--randomization
none`, `eval_checkpoint_tree.py` with the evaluator arguments of
`../2026-08-31-star-v2-curves` verbatim. One seed (0), one checkpoint.

`nophase`, `nophase_wd_clin`, `nophase_linlr`, `nophase` seed 1, and
`nophase_bias2x` were cancelled in the stall band and are not scored; the
user decision of 2026-09-01 is to keep the (constant at hold 1) sin/cos
phase pair.

## The 3B-only tree

`eval_checkpoint_tree.py` scores every milestone in a tree, so each arm gets
a tree holding only the 3B checkpoint, as a relative symlink into the training
tree (relative, so it resolves inside the container's `/data` bind):

```bash
ssh ice 'D=/home/hice1/fwu91/scratch/Research/IsaacLab/data
for a in z64_merged enc_hist obs_hist; do
  src=$(ls $D/latent64_probe_10b/${a}_seed0/tracker/*/models/model_step_3000238080.pt)
  dst=$D/latent64_probe_3b/${a}_seed0/tracker/f3000238080/models
  mkdir -p $dst
  ln -sfn "$(realpath --relative-to=$dst $src)" $dst/model_step_3000238080.pt
done'
```

## Submit

```bash
for arm in z64_merged enc_hist obs_hist; do
  pixi run python -m imitation_experiments.pipeline.cluster plan \
    --campaign experiments/campaigns/2026-09-01-latent64-probe-3b-eval/campaign.yaml \
    --arm $arm --seed 0
  # then the printed: ... submit --plan <dir> --confirm <PLAN_SHA>
done
```

Scored rows land in `/data/eval/latent64_probe_3b/<arm>_seed0_clean_f3000238080.json`
on ICE (`/home/hice1/fwu91/scratch/Research/IsaacLab/data/eval/latent64_probe_3b/`).

## Rows (2026-09-02)

Jobs 5607230 / 5607228 / 5607229 (5607227 died in the flaky Kit startup
crash and was resubmitted as 5607230). One seed, one checkpoint, preliminary.

| arm | SR | MPJPE-L | MPJPE-G | acc | jerk | action_delta | action_jerk |
|---|---:|---:|---:|---:|---:|---:|---:|
| `z64_merged` | 0.9102 | 23.89 | 108.82 | 4.42 | 181.5 | 0.754 | 0.701 |
| `enc_hist` | 0.9111 | 23.44 | 122.25 | 4.48 | 181.7 | 0.752 | 0.692 |
| `obs_hist` | 0.9026 | 25.09 | 94.74 | 5.08 | 226.1 | 0.908 | 0.954 |
| `sonic_v1_1` (4096 board) | 0.9888 | 26.73 | 187.7 | 3.45 | - | - | - |

## Inspection clips (2026-09-02)

`./render_clips.sh` renders the three 3B trackers on their own frozen-encoder
latents (the tracker ceiling, no planner), five ranks per arm chosen from the
eval JSONs so agreement and disagreement are both on screen (see the script
header), studio look, 1280x720, PhysX for the RTX camera. Inputs are the
workstation mirror `logs/latent64_probe_mirror/` (checkpoints, eval JSONs, the
`p5_concat` encoder) plus the star-v2 hub encoder mirror. Output:
`logs/latent64_probe_mirror/clips/<arm>_3B/videos/rank-<rank>-<motion>.mp4`
with stills every 50 steps. About four minutes per arm for all five clips.

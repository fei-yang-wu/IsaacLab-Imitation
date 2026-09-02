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

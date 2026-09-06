# 2026-09-06 -- variable-window-5b scored at 5B, plus the live-policy matrix

Scores the six `2026-09-05-variable-window-5b` trackers at their 5B
checkpoint on `bones_testbed4096_v1` clean (star-v2 curves protocol
verbatim), and the `combo` control at its matched 5B checkpoint.

## Rows

| row | what moves |
|---|---|
| `clean` | the arm's train live policy: padded / sequence / fixed_target at 10, stride at 1, nested / block at `base` |
| `live<m>` | same 5B tracker, same encoder, frozen encoder at member `m` of its set (`agent.ipmd.hl_skill_live_horizon=<m>`) |

Matrix members: padded / sequence / fixed_target {2, 5, 15}; stride {2, 3};
nested {2, 5, 10} (prefix truncation); block {2, 5, 10, 15} (one block kept).

Control `combo`: `latent64-probe-10b` tracker at `model_step_5000134656`
through a one-checkpoint mirror tree
(`/storage/ice-shared/vip-vwt/scratch-fwu91/eval_trees/variable_window_5b/combo_seed0/tracker/f5000134656/`,
symlink into `/data/latent64_probe_10b/...`). Its 10B endpoint row
(0.9214 / 22.64 / 88.52) is NOT the matched comparison.

Outputs: `/storage/ice-shared/vip-vwt/scratch-fwu91/eval/variable_window_5b/<arm>_seed0_<row>_f5000134656.json`.
25 jobs, one checkpoint each, 2 h limit.

```bash
./experiments/campaigns/2026-09-06-variable-window-5b-eval/submit.sh vw_padded 0
```

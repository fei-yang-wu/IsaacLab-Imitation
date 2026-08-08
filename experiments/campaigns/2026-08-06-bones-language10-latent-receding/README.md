# BONES language10 latent receding-horizon study

Status (2026-08-06): complete. The one-update H3 trainer smoke and a real
kit-less Newton rollout with exponential overlap both pass. Full
materialization retained 49,000 matched publications in each target frame,
dropped only the expected 2,900 trajectory-tail publications, and reproduced
frozen z0 within `2.384e-06`. Both 10k planners and all 70 closed-loop rollout
jobs completed, and the strict aggregate passed under
`logs/bones_language10_latent_receding_seed0`.

This focused ablation follows the 2026-08-05 trajectory-first oracle planner.
It predicts three ordered H10 latent tokens at 5 Hz and still executes one H10
command before replanning. The frozen root-qpos encoder, 5 Hz publication rate,
50 Hz tracker, language goals, low-level checkpoint, randomization, no-push
override, and 10k optimizer budget remain fixed.

The grid intentionally varies only two coupled questions:

| target frame | first token | raw exponential | clipped/gated exponential |
|---|:---:|:---:|:---:|
| future publication (transport-aware supervision) | yes | yes | yes |
| current publication (stale-frame diagnostic) | yes | yes | yes |

The existing H1 10k checkpoint is the seventh row. The exponential decay is
0.5, giving full-history weights approximately `[0.506, 0.307, 0.186]` from
freshest to oldest. Clipped/gated fusion clips each old residual to one training
standard deviation and rejects it above normalized RMS distance 2.0 or below
cosine agreement 0.5. Phase is generated after fusion and never averaged.

The stored trajectory keys provide future-publication targets without another
rollout: `(env_id, episode_id, planner_step + k)` selects token `k`. The
alternative current-publication target encodes the stored `[30,38]` root-qpos
window as three H10 packets in the original robot frame. Both materializations
require all 30 validity bits and use identical rows.

Run locally from the repository root:

```bash
STAGES=materialize experiments/campaigns/2026-08-06-bones-language10-latent-receding/run.sh
STAGES=train experiments/campaigns/2026-08-06-bones-language10-latent-receding/run.sh
STAGES=eval,aggregate experiments/campaigns/2026-08-06-bones-language10-latent-receding/run.sh
```

The 10 Hz execute-5 idea is not included in this grid. The current tracker was
trained for a ten-step held code and ten-step phase; refreshing it after five
steps would conflate planner cadence with a low-level train/deploy mismatch.
Qualify an oracle execute-5 tracker contract before treating cadence as another
comparison row.

## Result

| variant | target frame | execution | SONIC SR | successful MPJPE-L (mm) |
|---|---|---|---:|---:|
| H3 future | future publication | first | **0.401** | 49.82 |
| H3 future | future publication | clipped/gated | 0.396 | **39.72** |
| H3 future | future publication | exponential | 0.394 | 39.80 |
| H3 current | current publication | first | 0.359 | 50.24 |
| H1 control | H1 | first | 0.329 | 46.08 |
| H3 current | current publication | clipped/gated | 0.317 | 47.42 |
| H3 current | current publication | exponential | 0.283 | 45.49 |

The predeclared SR-first ranking selects future-publication H3 with fresh-only
execution. It improves the matched H1 control by 7.2 SR points. If completion
and local tracking quality are treated jointly, future-publication
clipped/gated overlap is the practical Pareto alternative: it gives up only
five successes in 1,000 while reducing success-only MPJPE-L by 10.10 mm.

The near-tied totals hide motion-specific routing effects. Relative to
future/fresh-only, clipped/gated overlap changes feeding-birds success from
77/100 to 2/100, but improves stoop from 9/100 to 38/100 and mosquito-drive-away
from 15/100 to 51/100. Do not claim one universal fusion rule from this seed;
use future-publication targets, retain both fresh-only and clipped/gated as the
two justified operating points, and replicate or learn an explicitly
goal-conditioned gate before selecting one for every motion. The stale-frame
overlap rows are rejected.

Audited outputs:

- `logs/bones_language10_latent_receding_seed0/aggregate/results.md`
- `logs/bones_language10_latent_receding_seed0/aggregate/results.json`

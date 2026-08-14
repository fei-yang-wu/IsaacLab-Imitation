# DiffSR latent failure analysis

This note summarizes the remaining failures in the existing SONIC-compatible
shared-4096 eval results for the DiffSR latent low-level tracker. Failure means
that a SONIC tracking-failure term fires: `anchor_pos`, `anchor_ori`, or
`ee_body_pos`. `reference_finished` is not a failure.

## Summary

At the final available checkpoint, 7.668B environment frames, DiffSR latent has
SR 0.9358, success-only MPJPE-L 25.28 mm, and 263 failed motions out of 4,096.

The remaining failures are mostly end-effector tracking failures:

| failure term | count | share of failed motions |
| --- | ---: | ---: |
| `ee_body_pos` | 216 | 82.1% |
| `anchor_ori` | 41 | 15.6% |
| `anchor_pos` | 7 | 2.7% |

One motion fired both `anchor_pos` and `ee_body_pos`, so term counts can sum to
more than the number of failed motions.

## Trend

| Environment Frames (B) | SR | MPJPE-L (mm) | failed | `ee_body_pos` | `anchor_ori` | `anchor_pos` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.359 | 0.9170 | 26.03 | 340 | 283 | 50 | 14 |
| 3.981 | 0.9265 | 25.17 | 301 | 253 | 44 | 10 |
| 5.997 | 0.9321 | 25.09 | 278 | 227 | 44 | 12 |
| 7.668 | 0.9358 | 25.28 | 263 | 216 | 41 | 7 |

The main improvement after 2.359B frames is the reduction in `ee_body_pos`
failures. `anchor_ori` improves more slowly and stays near 40-50 failed
motions. `anchor_pos` is already small.

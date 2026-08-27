# Figures

`main.tex` draws every figure through `\phfig{file}{height}{description}`.
While the preamble says `\figsreadyfalse`, each one renders as a labelled box
of the right footprint, so the paper compiles and paginates before any figure
exists. Drop the PDFs in this directory and flip the preamble switch to
`\figsreadytrue` to swap them all in at once. A single figure can be brought in
early by replacing its `\phfig` call with `\includegraphics`.

Keep filenames exactly as listed; the placeholder boxes print the filename they
expect.

| File | Where | Width | What it shows | Source |
| --- | --- | --- | --- | --- |
| `overview.pdf` | Sec. I, teaser | `figure*` | Encoder window, the low-rank diffusion objective, the code, the frozen tracker, the behavior model | hand-drawn schematic |
| `objective.pdf` | Sec. III-B | 1 col | The factorized noise predictor; window and code through one branch, noised target through the other, meeting in a rank-`d` inner product. Inset: the reconstruction baseline for contrast | hand-drawn schematic |
| `pareto.pdf` | Sec. IV-B | 1 col | Success rate against success-only MPJPE-L; one point per system and per design-study arm, Pareto front drawn | `logs/testbed4096/*.json` + `logs/interface_design_study_eval/*_clean_*.json` |
| `terminations.pdf` | Sec. IV-B | 1 col | Stacked termination cause (anchor height / anchor orientation / end-effector height) as a fraction of failed episodes | `termination_terms` field of the same board JSONs |
| `filmstrip_tracking.pdf` | Sec. IV-B | `figure*` | Reference-vs-robot filmstrip on three hard clips, ours over baseline | rendered rollouts; see `logs/testbed4096/failure_videos/` and `success_videos/` for the clip list |
| `curves.pdf` | Sec. IV-C | `figure*` | Three panels against environment frames: success rate by bottleneck axis, success rate by objective axis, MPJPE-G by objective axis | `logs/interface_design_study_eval/*_milestone_f*.json`, eight checkpoints per arm |
| `local_vs_global.pdf` | Sec. IV-C | 1 col | MPJPE-L against MPJPE-G, one point per arm, reconstruction and the hold-1 arms labelled | same milestone/endpoint rows |
| `cadence.pdf` | Sec. IV-D | 1 col | Tracking error against slots consumed, with published-to-oracle agreement on the second axis, latent and explicit routes | consumption sweep of `experiments/campaigns/2026-08-17-planner-10b-trackers/eval.sh` |
| `induced_error.pdf` | Sec. IV-D | 1 col | Paired ceiling / planner bars per interface with the induced-error gap annotated | oracle (`ORACLE=1`) and planner rows of the same campaign |
| `latent_structure.pdf` | Sec. IV-E | 1 col | (a) latent projection coloured by semantic region with a nearest-neighbour gallery, (b) retrieved-to-random kinematic distance against `k` with the bootstrap interval | `imitation_experiments.evaluation.analyze_cross_motion_latent_structure` and `build_latent_neighbor_gallery` |
| `filmstrip_chaining.pdf` | Sec. IV-E | `figure*` | One `A -> B` episode with the switch frame marked, its control run underneath, and both latent trajectories on the region map | `imitation_experiments.evaluation.eval_gr00t_chaining` |

## Rules for these panels

- Two-dimensional projections are display coordinates only. Every reported
  number comes from the metric space, never from the projection.
- Any figure that shows tracking error shows both frames, local and global, or
  says in its caption which one it is and why the other is not shown.
- A curve figure that stops before an arm has converged says so on the plot.
- Filmstrip frames are evenly sampled from one episode. Do not assemble a strip
  from several reruns.

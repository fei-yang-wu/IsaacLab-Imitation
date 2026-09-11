# Entrypoints

scripts/ contains command-line wiring; shared implementation belongs in
source/imitation_experiments/.

- rlopt/ and rsl_rl/: training, evaluation, and playback.
- data/, audit/, viz/, bench/: preparation, gates, rendering, and benchmarks.
- runtime_bootstrap.py: shared Isaac runtime setup in scripts/rlopt/.

Run from the repository root through Pixi. Isaac entrypoints use
pixi run -e isaaclab. Preserve --task selection and env./agent. Hydra overrides.
Entrypoints may change config after Hydra parsing; recorded resolved config
is the authority for a run. Print retained video paths as absolute paths.

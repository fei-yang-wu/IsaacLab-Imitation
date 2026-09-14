"""Restating a frame-sized recipe at the live environment count.

A released on-policy recipe states its rollout split and checkpoint cadence in
counts that hold at every environment count. RLOpt sizes both in frames, so a
translated config only reproduces the recipe at the environment count it was
written against. These tests pin the restatement.

`train_impl` imports Isaac Lab at module scope, so the helpers are loaded from
source here instead of importing the module.
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

_SOURCE = Path(__file__).with_name("train_impl.py")
_HELPERS = (
    "_reference_rollout_batch",
    "_declared_mini_batch_size",
    "_restated_frame_interval",
    "_restate_declared_intervals",
    "_step_counter_limit",
)

REFERENCE_NUM_ENVS = 4096
HORIZON = 24
MINI_BATCHES = 4
SAVE_ITERATIONS = 200
LOG_ITERATIONS = 100
REFERENCE_BATCH = REFERENCE_NUM_ENVS * HORIZON
REFERENCE_MINI_BATCH = REFERENCE_BATCH // MINI_BATCHES


def _load_helpers() -> dict:
    tree = ast.parse(_SOURCE.read_text())
    wanted = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in _HELPERS
    }
    missing = [name for name in _HELPERS if name not in wanted]
    if missing:
        raise AssertionError(f"train_impl.py is missing helpers: {missing}")

    class _SilentLogger:
        def warning(self, *_args, **_kwargs) -> None:
            return None

    namespace: dict = {"logger": _SilentLogger()}
    source = "\n".join(textwrap.dedent(ast.unparse(wanted[name])) for name in _HELPERS)
    exec(  # noqa: S102 - the source is this repo's own file
        "from __future__ import annotations\n" + source, namespace
    )
    return namespace


_NS = _load_helpers()
declared_mini_batch_size = _NS["_declared_mini_batch_size"]
restate_intervals = _NS["_restate_declared_intervals"]
step_counter_limit = _NS["_step_counter_limit"]


class _Trainer:
    """The trainer sub-config that owns the frame-counted log interval."""

    def __init__(self, log_interval: int) -> None:
        self.log_interval = log_interval


class _DeclaredCfg:
    """An agent config that declares the released recipe in counts."""

    def __init__(self) -> None:
        self.reference_num_envs = REFERENCE_NUM_ENVS
        self.mini_batches_per_rollout = MINI_BATCHES
        self.save_interval_iterations = SAVE_ITERATIONS
        self.log_interval_iterations = LOG_ITERATIONS
        self.save_interval = SAVE_ITERATIONS * REFERENCE_BATCH
        self.trainer = _Trainer(LOG_ITERATIONS * REFERENCE_BATCH)


class _UndeclaredCfg:
    """An agent config that states frame counts and declares no reference."""

    def __init__(self) -> None:
        self.save_interval = 999
        self.trainer = _Trainer(999)


def test_rollout_splits_into_the_declared_minibatch_count_at_every_env_count() -> None:
    for num_envs in (16, 128, 1024, REFERENCE_NUM_ENVS, 8192):
        batch = num_envs * HORIZON
        resolved = declared_mini_batch_size(
            _DeclaredCfg(),
            configured_mini_batch=REFERENCE_MINI_BATCH,
            per_env_horizon=HORIZON,
            scaled_frames_per_batch=batch,
        )
        assert resolved is not None
        assert batch // resolved == MINI_BATCHES, (
            f"num_envs={num_envs} split into {batch // resolved} minibatches"
        )


def test_reference_env_count_keeps_the_configured_minibatch_size() -> None:
    resolved = declared_mini_batch_size(
        _DeclaredCfg(),
        configured_mini_batch=REFERENCE_MINI_BATCH,
        per_env_horizon=HORIZON,
        scaled_frames_per_batch=REFERENCE_BATCH,
    )
    assert resolved == REFERENCE_MINI_BATCH


def test_an_explicit_minibatch_size_is_not_restated() -> None:
    assert (
        declared_mini_batch_size(
            _DeclaredCfg(),
            configured_mini_batch=REFERENCE_MINI_BATCH + 1,
            per_env_horizon=HORIZON,
            scaled_frames_per_batch=384,
        )
        is None
    )


def test_a_config_without_a_declaration_is_not_restated() -> None:
    assert (
        declared_mini_batch_size(
            _UndeclaredCfg(),
            configured_mini_batch=REFERENCE_MINI_BATCH,
            per_env_horizon=HORIZON,
            scaled_frames_per_batch=384,
        )
        is None
    )


def test_checkpoint_cadence_holds_at_every_env_count() -> None:
    for num_envs in (16, 128, 1024, REFERENCE_NUM_ENVS, 8192):
        batch = num_envs * HORIZON
        cfg = _DeclaredCfg()
        restate_intervals(cfg, per_env_horizon=HORIZON, scaled_frames_per_batch=batch)
        assert cfg.save_interval % batch == 0
        assert cfg.save_interval // batch == SAVE_ITERATIONS


def test_an_explicit_save_interval_is_not_restated() -> None:
    cfg = _DeclaredCfg()
    cfg.save_interval = 12_345
    restate_intervals(cfg, per_env_horizon=HORIZON, scaled_frames_per_batch=384)
    assert cfg.save_interval == 12_345


def test_logging_cadence_holds_at_every_env_count() -> None:
    """A short run must still emit metrics, not only a final summary line."""

    for num_envs in (16, 128, 1024, REFERENCE_NUM_ENVS, 8192):
        batch = num_envs * HORIZON
        cfg = _DeclaredCfg()
        restate_intervals(cfg, per_env_horizon=HORIZON, scaled_frames_per_batch=batch)
        assert cfg.trainer.log_interval // batch == LOG_ITERATIONS


def test_an_explicit_log_interval_is_not_restated() -> None:
    cfg = _DeclaredCfg()
    cfg.trainer.log_interval = 4_321
    restate_intervals(cfg, per_env_horizon=HORIZON, scaled_frames_per_batch=384)
    assert cfg.trainer.log_interval == 4_321


def test_a_config_without_a_declaration_keeps_its_save_interval() -> None:
    cfg = _UndeclaredCfg()
    restate_intervals(cfg, per_env_horizon=HORIZON, scaled_frames_per_batch=384)
    assert cfg.save_interval == 999
    assert cfg.trainer.log_interval == 999


class _Env:
    """An environment that declares its own episode length in control steps."""

    def __init__(self, max_episode_length) -> None:
        self.unwrapped = self
        self.max_episode_length = max_episode_length


def test_step_cap_clears_a_long_reference_episode() -> None:
    """A 24 s reference at half speed is 1005 control steps; the cap must clear it."""

    assert step_counter_limit(_Env(1005), 500) == 1006


def test_step_cap_keeps_the_fallback_for_short_episodes() -> None:
    """A short task keeps the historical cap, so existing runs are unchanged."""

    assert step_counter_limit(_Env(400), 500) == 500


def test_step_cap_falls_back_when_the_episode_length_is_unusable() -> None:
    for declared in (None, 0, -1, "many", float("nan")):
        assert step_counter_limit(_Env(declared), 500) == 500


def test_step_cap_accepts_an_environment_without_the_attribute() -> None:
    assert step_counter_limit(object(), 500) == 500

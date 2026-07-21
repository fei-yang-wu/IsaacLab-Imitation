from __future__ import annotations

from collections.abc import Sequence

from isaaclab_imitation.envs import ImitationRLEnv


def anneal_termination_threshold_by_frames(
    env: ImitationRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    start_value: float,
    end_value: float,
    start_frames: int,
    end_frames: int,
    param_name: str = "threshold",
) -> float:
    """Linearly anneal one termination-term parameter over collected frames.

    Single-GPU adaptation of the SONIC release protocol: the release trains
    with its strict thresholds from iteration zero at 64+ GPU scale, which is
    prohibitively slow locally. This term starts from a looser value (the
    release's own base/eval thresholds) and reaches the strict release value
    by ``end_frames``, so the final protocol - and every frame collected after
    ``end_frames`` - is identical to training without a curriculum.

    Frame count is ``common_step_counter * num_envs`` (control steps across
    all environments), making the schedule reproducible for a given env count
    and total-frame budget. The annealed value is returned so the curriculum
    manager logs it under ``Curriculum/<term_name>``.
    """
    del env_ids
    total_frames = int(env.common_step_counter) * int(env.num_envs)
    if end_frames <= start_frames:
        progress = 1.0
    else:
        progress = (total_frames - start_frames) / float(end_frames - start_frames)
        progress = min(max(progress, 0.0), 1.0)
    value = float(start_value) + (float(end_value) - float(start_value)) * progress
    term_cfg = env.termination_manager.get_term_cfg(term_name)
    term_cfg.params[param_name] = value
    return value

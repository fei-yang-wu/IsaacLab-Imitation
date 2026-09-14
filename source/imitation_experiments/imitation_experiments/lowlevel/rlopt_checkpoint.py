"""Attach Isaac curriculum counters to RLOpt's generic checkpoint hooks."""

from pathlib import Path

from iltools.core import sha256_file


def attach_isaac_training_state(agent, env) -> None:
    """Resume at new episodes with the saved curriculum clock and data identity.

    Network, optimizer, normalizer, and frame-budget state remain owned by
    RLOpt. This workspace hook supplies the environment-specific clock.
    """
    base = getattr(env, "unwrapped", env)
    if not hasattr(base, "common_step_counter"):
        return
    old_save = agent._extra_model_state_dict
    old_load = agent._load_extra_model_state_dict
    manager = getattr(base, "curriculum_manager", None)
    # Isaac CurriculumManager has no public term accessor. Capture supported
    # term hooks once, while the simulator is alive, including for final saves.
    stateful_terms = {
        name: cfg.func
        for name, cfg in zip(
            getattr(manager, "_term_names", []),
            getattr(manager, "_term_cfgs", []),
            strict=True,
        )
        if callable(getattr(cfg.func, "training_state_dict", None))
        and callable(getattr(cfg.func, "load_training_state_dict", None))
    }

    def fingerprint():
        cfg = getattr(base, "cfg", None)
        manifest = getattr(getattr(cfg, "reference", None), "manifest", "")
        manifest = manifest or getattr(cfg, "reference_manifest", "")
        return sha256_file(Path(manifest)) if manifest else None

    # TorchRL can close the simulator when it yields the final rollout.
    # Freeze immutable provenance while scene-backed properties still exist;
    # the plain curriculum counter remains readable after that close.
    immutable_state = {
        "schema": 1,
        "num_envs": int(base.num_envs),
        "reference_manifest_sha256": fingerprint(),
        "task": getattr(getattr(base, "spec", None), "id", None),
    }

    def save():
        state = old_save()
        state["isaaclab_training_state"] = {
            **immutable_state,
            "common_step_counter": int(base.common_step_counter),
            "curriculum_terms": {
                name: term.training_state_dict()
                for name, term in stateful_terms.items()
            },
        }
        return state

    def load(checkpoint):
        old_load(checkpoint)
        state = checkpoint.get("isaaclab_training_state")
        if state is None:
            return
        if state.get("schema") != 1 or int(state["common_step_counter"]) < 0:
            raise ValueError("Unsupported Isaac training-state checkpoint.")
        if state.get("reference_manifest_sha256") != fingerprint():
            raise ValueError(
                "Checkpoint Reference differs from the current training data."
            )
        task = getattr(getattr(base, "spec", None), "id", None)
        if task != state.get("task"):
            raise ValueError("Checkpoint task differs from the current task.")
        saved_terms = state.get("curriculum_terms", {})
        if set(saved_terms) != set(stateful_terms):
            raise ValueError("Stateful curriculum terms differ from the checkpoint.")
        for name, term in stateful_terms.items():
            term.load_training_state_dict(saved_terms[name])
        base.common_step_counter = int(state["common_step_counter"])
        curriculum = getattr(base, "curriculum_manager", None)
        if curriculum is not None:
            curriculum.compute(env_ids=None)

    agent._extra_model_state_dict = save
    agent._load_extra_model_state_dict = load

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections import deque
from collections.abc import Mapping

import gymnasium as gym
import torch
from rlopt.agent import FastSACRLOptConfig, FastTD3RLOptConfig, IPMDBilinearRLOptConfig, IPMDRLOptConfig, IPMDSRRLOptConfig, PPORLOptConfig, SACRLOptConfig  # noqa: F401
from rlopt.config_base import RLOptConfig  # noqa: F401
from torchrl.data.tensor_specs import Composite, Unbounded
from torchrl.envs.libs.gym import (
    GymWrapper,
    _gym_to_torchrl_spec_transform,
    terminal_obs_reader,
)
from tensordict import TensorDict
from tensordict.base import TensorDictBase


class IsaacLabWrapper(GymWrapper):
    """A wrapper for IsaacLab environments.

    Args:
        env (isaaclab.envs.ManagerBasedRLEnv or equivalent): the environment instance to wrap.
        categorical_action_encoding (bool, optional): if ``True``, categorical
            specs will be converted to the TorchRL equivalent (:class:`torchrl.data.Categorical`),
            otherwise a one-hot encoding will be used (:class:`torchrl.data.OneHot`).
            Defaults to ``False``.
        allow_done_after_reset (bool, optional): if ``True``, it is tolerated
            for envs to be ``done`` just after :meth:`reset` is called.
            Defaults to ``False``.

    For other arguments, see the :class:`torchrl.envs.GymWrapper` documentation.

    Refer to `the Isaac Lab documentation <https://isaac-sim.github.io/IsaacLab/>`_.

    Example:
        >>> # This code block ensures that the Isaac app is started in headless mode
        >>> from scripts_isaaclab.app import AppLauncher
        >>> import argparse

        >>> parser = argparse.ArgumentParser(description="Train an RL agent with TorchRL.")
        >>> AppLauncher.add_app_launcher_args(parser)
        >>> args_cli, hydra_args = parser.parse_known_args(["--headless"])
        >>> app_launcher = AppLauncher(args_cli)

        >>> # Imports and env
        >>> import gymnasium as gym
        >>> import isaaclab_tasks  # noqa: F401
        >>> from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg
        >>> from torchrl.envs.libs.isaac_lab import IsaacLabWrapper

        >>> env = gym.make("Isaac-Ant-v0", cfg=AntEnvCfg())
        >>> env = IsaacLabWrapper(env)

    """

    def __init__(
        self,
        env: ManagerBasedRLEnv,  # noqa: F821
        *,
        categorical_action_encoding: bool = False,
        allow_done_after_reset: bool = True,
        convert_actions_to_numpy: bool = False,
        device: torch.device | None = None,
        **kwargs,
    ):
        if device is None:
            device = torch.device("cuda:0")
        super().__init__(
            env,
            device=device,
            categorical_action_encoding=categorical_action_encoding,
            allow_done_after_reset=allow_done_after_reset,
            convert_actions_to_numpy=convert_actions_to_numpy,
            **kwargs,
        )
        # Keep only the latest log payload to avoid retaining large per-step
        # info dicts (often CUDA tensors) across long rollouts.
        self.log_infos = deque(maxlen=1)

    def _base_isaac_env(self):
        env = getattr(self, "_env", None)
        return getattr(env, "unwrapped", env)

    def sample_expert_batch(self, batch_size: int, required_keys):
        return self._base_isaac_env().sample_expert_batch(
            batch_size=batch_size,
            required_keys=required_keys,
        )

    def set_agent_latent_command(self, latent_command, env_ids=None):
        return self._base_isaac_env().set_agent_latent_command(
            latent_command,
            env_ids=env_ids,
        )

    def reset_agent_latent_command(self, env_ids=None):
        return self._base_isaac_env().reset_agent_latent_command(env_ids=env_ids)

    def get_agent_latent_command(self, env_ids=None):
        return self._base_isaac_env().get_agent_latent_command(env_ids=env_ids)

    @property
    def _is_batched(self) -> bool:
        return True

    def seed(self, seed: int | None):
        self._set_seed(seed)

    @staticmethod
    def _extract_log_info(
        info: dict[str, torch.Tensor | float | int | str | bool | None],
    ) -> dict[str, object]:
        """Extract a compact, logging-only info payload with CPU plain values.

        IsaacLab info dicts can include large entries (e.g., final_obs). For
        metrics logging we only need the ``log`` subtree, and we must ensure it
        does not retain CUDA tensors to avoid illegal memory access once the
        underlying device buffers are freed.
        """

        for key, value in info.items():
            if isinstance(value, torch.Tensor):
                info[key] = value.detach().cpu().item()
            elif isinstance(value, Mapping):
                info[key] = IsaacLabWrapper._extract_log_info(value)
        return info

    def _build_env(
        self,
        env,
        from_pixels: bool = False,
        pixels_only: bool = False,
    ) -> gym.core.Env:  # noqa: F821
        env = super()._build_env(
            env,
            from_pixels=from_pixels,
            pixels_only=pixels_only,
        )
        env.autoreset_mode = "SameStep"
        return env

    def _make_specs(self, env: gym.Env, batch_size=None) -> None:  # noqa: F821
        # Build specs from IsaacLab's unbatched spaces to preserve observation keys.
        if batch_size is None:
            batch_size = self.batch_size
        env_unwrapped = getattr(env, "unwrapped", env)

        action_space = getattr(env_unwrapped, "single_action_space", None)
        action_needs_batch = action_space is not None
        action_space = action_space if action_space is not None else env.action_space
        action_spec = _gym_to_torchrl_spec_transform(
            action_space,
            device=self.device,
            categorical_action_encoding=self._categorical_action_encoding,
        )
        if action_needs_batch:
            action_spec = action_spec.expand(*batch_size, *action_spec.shape)  # type: ignore
        obs_space = getattr(env_unwrapped, "single_observation_space", None)
        obs_needs_batch = obs_space is not None
        obs_space = obs_space if obs_space is not None else env.observation_space
        observation_spec = _gym_to_torchrl_spec_transform(
            obs_space,
            device=self.device,
            categorical_action_encoding=self._categorical_action_encoding,
        )
        if obs_needs_batch:
            observation_spec = observation_spec.expand(
                *batch_size, *observation_spec.shape
            )  # type: ignore
        if not isinstance(observation_spec, Composite):
            if self.from_pixels:
                observation_spec = Composite(pixels=observation_spec, shape=batch_size)  # type: ignore
            else:
                observation_spec = Composite(
                    observation=observation_spec, shape=batch_size
                )  # type: ignore

        reward_space = self._reward_space(env)
        if reward_space is not None:
            reward_spec = _gym_to_torchrl_spec_transform(
                reward_space,
                device=self.device,
                categorical_action_encoding=self._categorical_action_encoding,
            )
        else:
            reward_spec = Unbounded(shape=[1], device=self.device).expand(
                *batch_size, 1
            )  # type: ignore
        if reward_space is not None:
            reward_spec = reward_spec.expand(*batch_size, *reward_spec.shape)  # type: ignore

        self.done_spec = self._make_done_spec()  # type: ignore
        self.action_spec = action_spec  # type: ignore
        self.reward_spec = reward_spec  # type: ignore
        self.observation_spec = observation_spec  # type: ignore

    def _output_transform(self, step_outputs_tuple):  # type: ignore
        # IsaacLab will modify the `terminated` and `truncated` tensors
        #  in-place. We clone them here to make sure data doesn't inadvertently get modified.
        # The variable naming follows torchrl's convention here.
        observations, reward, terminated, truncated, info = step_outputs_tuple
        self.log_infos.append(self._extract_log_info(info["log"]))

        done = terminated | truncated

        # IsaacLab emits Gymnasium-style keys: final_obs / final_info.
        # Keep only terminal entries to avoid introducing scalar log info into
        # the tensordict info path.
        if isinstance(info, dict) and "final_obs" in info:
            info = {"final_obs": info["final_obs"]}
            return (
                observations,
                reward.unsqueeze(-1),
                terminated.clone().to(dtype=torch.bool),
                truncated.clone().to(dtype=torch.bool),
                done.clone().to(dtype=torch.bool),
                info,
            )
        else:
            return (
                observations,
                reward.unsqueeze(-1),
                terminated.clone().to(dtype=torch.bool),
                truncated.clone().to(dtype=torch.bool),
                done.clone().to(dtype=torch.bool),
                {},
            )

    def _reset_output_transform(self, reset_data):
        """Transform the output of the reset method."""
        observations, info = reset_data
        self.log_infos.append(self._extract_log_info(info["log"]))
        return (observations, {})

    @staticmethod
    def _normalize_nested_batch_sizes(td: TensorDictBase) -> None:
        """Recursively align nested TensorDict batch sizes with their parent."""
        parent_batch = td.batch_size
        for key in td.keys():
            value = td.get(key)
            if not isinstance(value, TensorDictBase):
                continue
            current = value
            if current.batch_size != parent_batch:
                current = current.clone()
                current.batch_size = parent_batch
                td.set(key, current)
            IsaacLabWrapper._normalize_nested_batch_sizes(current)

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        td_out = super()._step(tensordict)
        self._normalize_nested_batch_sizes(td_out)
        return td_out

    def _reset(self, tensordict: TensorDictBase | None = None, **kwargs) -> TensorDict:
        td_out = super()._reset(tensordict, **kwargs)
        self._normalize_nested_batch_sizes(td_out)
        return td_out


def CloneObsBuf(
    obs_buf: torch.Tensor | Mapping[str, object] | TensorDictBase,
) -> torch.Tensor | dict[str, object]:
    """Clone nested observation structures while normalizing TensorDict groups to dicts."""
    if isinstance(obs_buf, torch.Tensor):
        return obs_buf.clone()
    if isinstance(obs_buf, TensorDictBase):
        # Convert TensorDict groups to plain dicts so GymWrapper read_obs() can
        # re-encode them with Composite spec batch semantics.
        return {k: CloneObsBuf(obs_buf.get(k)) for k in obs_buf.keys()}
    if isinstance(obs_buf, Mapping):
        return {k: CloneObsBuf(v) for k, v in obs_buf.items()}
    return obs_buf


def CheckObsBufForNaN(
    obs_buf: dict[str, torch.Tensor | dict], prefix: str = ""
) -> None:
    """Recursively check nested observation dicts for NaNs."""
    for k, v in obs_buf.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            CheckObsBufForNaN(v, key)
        elif isinstance(v, torch.Tensor) and torch.isnan(v).any():
            first_row = v[0] if v.ndim > 0 else v
            print(
                f"NaN values found in observation {key} during step. First row: {first_row}"
            )
            raise ValueError(
                f"NaN values found in observation {key} during step. "
                "This is likely due to an error in the environment or the model."
            )


class IsaacLabTerminalObsReader(terminal_obs_reader):
    """A terminal observation reader for IsaacLab environments.

    This reader extracts the terminal observation from the environment's info dictionary.
    It is used to read the terminal observation when the environment is reset."""

    def __init__(self, observation_spec: Composite, backend, name: str = "final"):
        super().__init__(observation_spec=observation_spec, backend=backend, name=name)
        # Avoid recursive final/final specs when the wrapped observation spec
        # already exposes a terminal group.
        if name in self._obs_spec.keys():
            self._obs_spec.pop(name)
        # Provide info specs upfront to avoid dummy rollouts in set_info_dict_reader.
        self._info_spec = Composite({self.name: self._obs_spec.clone()}, shape=[])

    @staticmethod
    def _extract_nested_value(obs: object, key_path: tuple[str, ...]) -> object | None:
        cur = obs
        for key in key_path:
            if isinstance(cur, dict):
                if key not in cur:
                    return None
                cur = cur[key]
            else:
                # Scalar/tensor observation: only valid for single-key paths.
                return cur if len(key_path) == 1 else None
        return cur

    def _build_spec_buffer(
        self,
        spec: object,
        per_env_obs: list[object | None],
        key_path: tuple[str, ...],
    ) -> object:
        if isinstance(spec, Composite):
            td = spec.zero()
            for subkey in spec.keys():
                child = self._build_spec_buffer(
                    spec[subkey], per_env_obs, (*key_path, subkey)
                )
                td.set(subkey, child)
            return td

        buf = spec.zero()
        device = buf.device
        num_envs = len(per_env_obs)
        for i in range(num_envs):
            obs = per_env_obs[i]
            if obs is None:
                continue
            val = self._extract_nested_value(obs, key_path)
            if val is None:
                continue
            if isinstance(val, torch.Tensor):
                buf[i] = val.to(device=device)
            else:
                buf[i] = torch.as_tensor(val, device=device)
        return buf

    def __call__(self, info_dict, tensordict):
        # IsaacLab: info_dict["final_obs"] is np.ndarray(num_envs, dtype=object);
        # each entry is None or a nested dict produced by _slice_obs.
        backend_key = self.backend_key[self.backend]
        final_obs_arr = info_dict.pop(backend_key, None)
        info_dict.pop(self.backend_info_key[self.backend], None)

        # We intentionally skip parent terminal_obs_reader.__call__ here to
        # avoid recursive `final` spec self-merging. The IsaacLab wrapper
        # passes through only terminal observation info for this reader.

        num_envs = (
            len(final_obs_arr)
            if final_obs_arr is not None
            else int(tensordict.batch_size[0] if len(tensordict.batch_size) > 0 else 0)
        )
        per_env_obs: list[object | None] = [None] * num_envs
        if final_obs_arr is not None:
            for i in range(num_envs):
                per_env_obs[i] = final_obs_arr[i]

        for key in self.info_spec[self.name].keys():
            spec = self.info_spec[self.name, key]
            buf = self._build_spec_buffer(spec, per_env_obs, (key,))
            tensordict.set((self.name, key), buf)

        return tensordict

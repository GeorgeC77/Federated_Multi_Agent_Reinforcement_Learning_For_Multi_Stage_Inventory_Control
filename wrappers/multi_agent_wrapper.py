import numpy as np
import gymnasium as gym
from gymnasium import spaces

from envs.serial_multi_stage import SerialMultiStageEnv


class MultiAgentSerialWrapper(gym.Env):
    """
    Multi-agent wrapper for the SerialMultiStageEnv.

    - Each stage in the supply chain is treated as an independent agent:
      agent IDs = ["stage_0", "stage_1", ..., "stage_{n-1}"].

    - Observations, actions, and rewards follow the
      dictionary-based multi-agent API style, which is compatible
      with most MARL frameworks and federated RL implementations.

    Default:
      All agents receive a shared global reward (fully cooperative task).
      If needed, this wrapper can be extended to provide per-stage local rewards
      or shaped rewards.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, env: SerialMultiStageEnv):
        super().__init__()
        assert isinstance(env, SerialMultiStageEnv)
        self.env = env
        self.n_stages = env.n_stages
        self.agent_ids = [f"stage_{i}" for i in range(self.n_stages)]
        self.render_mode = env.render_mode

        # ----- Define multi-agent observation and action spaces -----
        # Each agent observes only its own per-stage observation
        per_dim = env.per_stage_obs_dim
        high_obs = np.full(per_dim, np.inf, dtype=np.float32)

        self.observation_space = spaces.Dict(
            {
                agent_id: spaces.Box(
                    low=-high_obs,
                    high=high_obs,
                    dtype=np.float32,
                )
                for agent_id in self.agent_ids
            }
        )

        # Each agent chooses a single continuous action: order quantity
        self.action_space = spaces.Dict(
            {
                agent_id: spaces.Box(
                    low=0.0,
                    high=env.max_order,
                    shape=(1,),
                    dtype=np.float32,
                )
                for agent_id in self.agent_ids
            }
        )

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        obs_dict = self._split_obs(obs)

        # For multi-agent API, return a dict of info for each agent
        info_dict = {agent_id: info.copy() for agent_id in self.agent_ids}
        return obs_dict, info_dict

    def step(self, action_dict):
        """
        action_dict example:
            {
                "stage_0": [a0],
                "stage_1": [a1],
                ...
            }

        Internally we convert the multi-agent dict into a joint action vector,
        then pass it to the underlying single-environment implementation.
        """

        # 1) Convert dict of local agent actions → joint action array
        joint_action = np.zeros(self.n_stages, dtype=np.float32)
        for i, agent_id in enumerate(self.agent_ids):
            a = np.asarray(action_dict[agent_id], dtype=np.float32).reshape(-1)
            joint_action[i] = a[0]

        # 2) Step the underlying serial supply chain environment
        obs, reward, terminated, truncated, info = self.env.step(joint_action)

        # 3) Split the global observation into per-agent observations
        obs_dict = self._split_obs(obs)

        # 4) Assign rewards
        # Default: each agent receives its own element from reward[i]
        # (where reward should be an array-like from the underlying environment)
        # If reward is a scalar global reward, modify here accordingly.
        rewards = {agent_id: reward[i] for i, agent_id in enumerate(self.agent_ids)}

        # 5) Termination flags for each agent
        terminateds = {agent_id: terminated for agent_id in self.agent_ids}
        truncateds = {agent_id: truncated for agent_id in self.agent_ids}

        # 6) Per-agent info
        infos = {agent_id: info.copy() for agent_id in self.agent_ids}

        return obs_dict, rewards, terminateds, truncateds, infos

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _split_obs(self, obs: np.ndarray) -> dict:
        """
        Convert the global observation vector from the single-environment
        into per-agent observations.

        Global obs shape:
            (n_stages * per_stage_obs_dim,)

        Per-stage obs:
            obs[i] = global[start:end] where
            start = i * per_stage_obs_dim
        """
        obs = np.asarray(obs, dtype=np.float32)
        per_dim = self.env.per_stage_obs_dim
        obs_dict = {}

        for i, agent_id in enumerate(self.agent_ids):
            start = i * per_dim
            end = start + per_dim
            obs_i = obs[start:end]
            obs_dict[agent_id] = obs_i

        return obs_dict

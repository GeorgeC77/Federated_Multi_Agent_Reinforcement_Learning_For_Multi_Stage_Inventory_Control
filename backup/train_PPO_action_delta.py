import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from envs.serial_multi_stage import SerialMultiStageEnv
from utils.utils import RunningNormalizer


# ==============================
#  Actor / Critic network
# ==============================

class ActorNet(nn.Module):
    """
    输出 Δa_t ∈ [-delta_max, delta_max]
    真实动作: a_t = clip(a_{t-1} + Δa_t, 0, max_action)
    """
    def __init__(self, obs_dim, action_dim=1, max_action=20.0, delta_fraction=0.5):
        super().__init__()
        self.max_action = max_action
        # 每步最大调整幅度，占 max_action 的比例，设得小一点防止大跳
        self.delta_max = delta_fraction * max_action

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )

        self.mean_head = nn.Linear(64, action_dim)
        # 全局 log_std（也可以改成 state-dependent）
        self.log_std = nn.Parameter(torch.ones(action_dim) * 0.5)

    def forward(self, x):
        h = self.net(x)
        mean = self.mean_head(h)      # pre-tanh mean（任意实数）
        std = self.log_std.exp()
        return mean, std

    def sample(self, obs):
        """
        训练用：从策略分布中采样 Δa_t + log_prob，用于 PPO。
        """
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)

        raw_action = dist.rsample()          # pre-tanh
        squashed = torch.tanh(raw_action)    # [-1, 1]
        delta_action = squashed * self.delta_max

        # tanh 修正
        log_prob = dist.log_prob(raw_action) - torch.log(1 - squashed.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1)

        return delta_action, log_prob

    def act_deterministic(self, obs):
        """
        评估用：用均值动作（不采样）得到 Δa_t（无 log_prob）。
        obs: torch.tensor, shape = (obs_dim,)
        """
        with torch.no_grad():
            h = self.net(obs)
            mean = self.mean_head(h)         # pre-tanh
            squashed = torch.tanh(mean)      # [-1, 1]
            delta_action = squashed * self.delta_max  # [-delta_max, delta_max]
        return delta_action.squeeze(-1)      # 标量（或者 (1,)）


class CriticNet(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # (batch,)


# ==============================
#  Helper: split obs by stage
# ==============================

def split_obs_for_stage(global_obs, n_stages, per_stage_obs_dim, stage_id):
    """
    global_obs: np.array or 1D tensor, shape = (n_stages * per_stage_obs_dim,)
    Return the local obs vector for stage_id
    """
    start = stage_id * per_stage_obs_dim
    end = start + per_stage_obs_dim
    return global_obs[start:end]


# ==============================
#  Evaluation: deterministic policy（用归一化 obs）
# ==============================

def evaluate_policy(
    n_stages,
    lead_time,
    actors,
    obs_normalizer,
    episode_length=100,
    n_eval_episodes=10,
):
    """
    用确定性策略（mean 动作）评估当前 actors 的表现。
    - env 仍然用物理量
    - 策略只看经过 obs_normalizer 归一化后的 obs
    返回：平均每 episode 的真实总 reward（可和 base-stock 对比）。
    """
    env = SerialMultiStageEnv(
        n_stages=n_stages,
        lead_times=[lead_time] * n_stages,
        episode_length=episode_length,
        render_mode=None,
    )

    per_stage_obs_dim = env.per_stage_obs_dim
    returns = []

    for ep in range(n_eval_episodes):
        global_obs, info = env.reset()
        # 评估时不再更新 normalizer，只用已有统计量
        norm_global_obs = obs_normalizer.normalize(global_obs)

        done = False
        truncated = False
        ep_return = 0.0

        # 每个 stage 当前的真实动作 a_t
        current_actions_env = [0.0 for _ in range(n_stages)]

        while not (done or truncated):
            actions_env = []

            for sid in range(n_stages):
                obs_sid = split_obs_for_stage(
                    norm_global_obs, n_stages, per_stage_obs_dim, sid
                )
                obs_sid_t = torch.tensor(obs_sid, dtype=torch.float32)

                # 确定性 Δa_t（用 mean）
                delta_sid = actors[sid].act_deterministic(obs_sid_t)
                delta_val = float(delta_sid.item())

                max_a = actors[sid].max_action
                new_action = np.clip(
                    current_actions_env[sid] + delta_val,
                    0.0,
                    max_a
                )
                current_actions_env[sid] = new_action
                actions_env.append(new_action)

            joint_action = np.array(actions_env, dtype=np.float32)
            next_global_obs, reward, done, truncated, info = env.step(joint_action)

            ep_return += float(np.sum(reward))

            # 用训练时的 normalizer 做归一化（不更新统计量）
            norm_global_obs = obs_normalizer.normalize(next_global_obs)
            global_obs = next_global_obs

        returns.append(ep_return)

    return float(np.mean(returns))


# ==============================
#  Main Training: PPO with Δ-action + obs 归一化 + smooth reg
# ==============================

def train_federated_actor_critic(
    n_stages=2,
    n_rounds=100,
    episodes_per_round=10,
    gamma=0.99,
    gae_lambda=0.97,
    ppo_epochs=10,
    ppo_clip=0.2,
    entropy_coef=0.01,
    lambda_smooth=5e-3,   # Δa 的平滑正则强度
    lead_time=2,          # 用于 env / eval
):
    # 1. Create environment (single serial supply chain)
    env = SerialMultiStageEnv(
        n_stages=n_stages,
        lead_times=[lead_time] * n_stages,
        episode_length=100,
        render_mode=None,
    )

    per_stage_obs_dim = env.per_stage_obs_dim
    global_obs_dim = env.observation_space.shape[0]

    # === 1.5 全局观测归一化器 ===
    # 假设 RunningNormalizer 接口为：
    #   norm = RunningNormalizer(dim)
    #   norm.update(x: np.ndarray)
    #   norm.normalize(x: np.ndarray) -> np.ndarray
    obs_normalizer = RunningNormalizer(global_obs_dim)

    # 2. Create an actor for each stage
    actors = []
    actor_optimizers = []
    for i in range(n_stages):
        actor = ActorNet(obs_dim=per_stage_obs_dim, action_dim=1)
        actors.append(actor)
        actor_optimizers.append(optim.Adam(actor.parameters(), lr=3e-4))

    # 3. Global critic (federated-averaged)
    global_critic = CriticNet(obs_dim=global_obs_dim)
    critic_lr = 3e-4

    for rnd in range(n_rounds):
        print(f"\n===== Federated Round {rnd+1}/{n_rounds} =====")

        local_critic_params_list = []
        round_rewards_all_steps = []

        # Each stage = one FL client
        for stage_id in range(n_stages):
            # 3.1 Clone global critic
            local_critic = copy.deepcopy(global_critic)
            critic_optimizer = optim.Adam(local_critic.parameters(), lr=critic_lr)

            # 3.2 Collect trajectories for this stage
            obs_buffer = []
            action_buffer = []   # 存 Δa_t
            logprob_buffer = []
            reward_buffer = []
            value_buffer = []
            done_buffer = []

            for ep in range(episodes_per_round):
                global_obs, info = env.reset()

                # 更新 normalizer 的统计量，并得到归一化后的 obs
                obs_normalizer.update(global_obs)
                norm_global_obs = obs_normalizer.normalize(global_obs)

                done = False
                truncated = False

                # 每个 episode 中，每个 stage 当前的真实动作 a_t
                current_actions_env = [0.0 for _ in range(n_stages)]

                while not (done or truncated):
                    actions_env = []   # 真实 a_t
                    log_probs_tmp = []
                    deltas_tmp = []    # Δa_t（tensor）

                    for sid in range(n_stages):
                        obs_sid = split_obs_for_stage(
                            norm_global_obs, n_stages, per_stage_obs_dim, sid
                        )
                        obs_sid_t = torch.tensor(obs_sid, dtype=torch.float32)

                        # 策略输出 Δa_t（有探索）
                        delta_sid, logp_sid = actors[sid].sample(obs_sid_t)
                        delta_val = float(delta_sid.detach().numpy()[0])

                        # a_t = clip(a_{t-1} + Δa_t, 0, max_action)
                        max_a = actors[sid].max_action
                        new_action = np.clip(
                            current_actions_env[sid] + delta_val,
                            0.0,
                            max_a
                        )
                        current_actions_env[sid] = new_action

                        actions_env.append(new_action)
                        log_probs_tmp.append(logp_sid)
                        deltas_tmp.append(delta_sid)

                    joint_action = np.array(actions_env, dtype=np.float32)

                    next_global_obs, reward, done, truncated, info = env.step(joint_action)

                    global_reward = np.sum(reward)

                    # critic 也看归一化之后的全局状态
                    value = local_critic(torch.tensor(norm_global_obs, dtype=torch.float32))

                    # 当前 stage 的局部观测（归一化后的 global obs 切片）
                    local_obs = split_obs_for_stage(
                        norm_global_obs, n_stages, per_stage_obs_dim, stage_id
                    )
                    local_obs_t = torch.tensor(local_obs, dtype=torch.float32)

                    # 缓存当前 stage 的轨迹
                    obs_buffer.append(local_obs_t)
                    action_buffer.append(deltas_tmp[stage_id].detach())  # Δa_t
                    logprob_buffer.append(log_probs_tmp[stage_id])       # log π(Δa_t)
                    reward_buffer.append(torch.tensor(global_reward, dtype=torch.float32))
                    value_buffer.append(value)
                    done_buffer.append(done or truncated)

                    round_rewards_all_steps.append(global_reward)

                    # 更新 obs / norm_obs
                    obs_normalizer.update(next_global_obs)
                    norm_next_global_obs = obs_normalizer.normalize(next_global_obs)
                    global_obs = next_global_obs
                    norm_global_obs = norm_next_global_obs

            # ============ 3.3 PPO UPDATE for this stage ============

            obs_tensor = torch.stack(obs_buffer)          # (T, obs_dim), 已是归一化 obs
            actions_tensor = torch.stack(action_buffer)   # (T, 1)   Δa_t
            old_logprobs_tensor = torch.stack(logprob_buffer).detach()  # (T,)
            rewards_tensor = torch.stack(reward_buffer)   # (T,)
            values_tensor = torch.stack(value_buffer)     # (T,) critic outputs
            dones_tensor = torch.tensor(done_buffer, dtype=torch.float32)  # (T,)

            T_steps = rewards_tensor.shape[0]

            # ---------- 标准 GAE（直接用原始 rewards_tensor） ----------
            advantages = torch.zeros(T_steps)
            returns = torch.zeros(T_steps)

            next_value = 0.0
            next_adv = 0.0

            values_detached = values_tensor.detach()

            for t in reversed(range(T_steps)):
                if dones_tensor[t] > 0.5:
                    next_value = 0.0
                    next_adv = 0.0

                delta = rewards_tensor[t] + gamma * next_value - values_detached[t]
                next_adv = delta + gamma * gae_lambda * next_adv

                advantages[t] = next_adv
                returns[t] = advantages[t] + values_detached[t]
                next_value = values_detached[t]

            # 标准化优势
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            returns = returns.detach()

            # ---------- PPO Optimization (multiple epochs on same batch) ----------
            delta_max = actors[stage_id].delta_max
            eps = 1e-6

            for _ in range(ppo_epochs):
                # 当前 actor 前向（输入已归一化）
                mean, std = actors[stage_id].forward(obs_tensor)
                dist = torch.distributions.Normal(mean, std)

                # actions_tensor 是 Δa_t ∈ [-delta_max, delta_max]
                # 反推 raw_action = atanh(Δa_t / delta_max)
                squashed = actions_tensor / delta_max      # [-1,1]
                squashed = torch.clamp(squashed, -1 + eps, 1 - eps)
                one = torch.ones_like(squashed)
                raw_action = 0.5 * torch.log((one + squashed) / (one - squashed))  # atanh

                logp_raw = dist.log_prob(raw_action)
                logp = logp_raw - torch.log(1 - squashed.pow(2) + 1e-6)
                logp = logp.sum(-1)   # (T,)

                entropy = dist.entropy().sum(-1).mean()

                # PPO ratio
                ratio = torch.exp(logp - old_logprobs_tensor)

                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - ppo_clip, 1 + ppo_clip) * advantages

                # Δa 的平滑正则 —— 抑制极端变化 / bang-bang
                smooth_penalty = (actions_tensor ** 2).mean()

                actor_loss = -torch.min(surr1, surr2).mean() \
                             - entropy_coef * entropy \
                             + lambda_smooth * smooth_penalty

                actor_optimizers[stage_id].zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actors[stage_id].parameters(), max_norm=1.0)
                actor_optimizers[stage_id].step()

            # ---------- Critic update ----------
            critic_loss = ((values_tensor - returns) ** 2).mean()
            critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(local_critic.parameters(), max_norm=1.0)
            critic_optimizer.step()

            # 3.4 收集本 stage 的 local critic 参数
            local_critic_params_list.append(copy.deepcopy(local_critic.state_dict()))

        # 3.5 FedAvg 聚合 critic
        print("Server: FedAvg critic parameters.")
        new_state_dict = copy.deepcopy(local_critic_params_list[0])
        for key in new_state_dict.keys():
            for i in range(1, len(local_critic_params_list)):
                new_state_dict[key] += local_critic_params_list[i][key]
            new_state_dict[key] /= float(len(local_critic_params_list))
        global_critic.load_state_dict(new_state_dict)

        # 训练阶段的 step-avg reward（带探索噪声的）
        if len(round_rewards_all_steps) > 0:
            avg_reward = np.mean(round_rewards_all_steps)
        else:
            avg_reward = 0.0
        print(f"Round {rnd+1}: approx avg step reward (stochastic, training) = {avg_reward:.2f}")

        # 每隔若干 round 做一次 deterministic 评估（同一个 normalizer）
        if (rnd + 1) % 10 == 0:
            eval_return = evaluate_policy(
                n_stages=n_stages,
                lead_time=lead_time,
                actors=actors,
                obs_normalizer=obs_normalizer,
                episode_length=100,
                n_eval_episodes=10,
            )
            print(f"[Eval deterministic] Round {rnd+1}: "
                  f"avg episode return = {eval_return:.2f}")


if __name__ == "__main__":
    # 先从最简单的情况开始：单 stage
    train_federated_actor_critic(
        n_stages=1,
        n_rounds=1000,
        episodes_per_round=100,
        lead_time=2,
    )

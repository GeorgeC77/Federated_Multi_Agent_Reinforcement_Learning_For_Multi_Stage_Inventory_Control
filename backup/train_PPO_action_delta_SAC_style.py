import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim

from envs.serial_multi_stage import SerialMultiStageEnv
from utils.utils import RunningNormalizer   # 如果不想用归一化，可以把相关调用注释掉


# ==============================
#  SAC-style Actor for PPO
# ==============================

class PPOActorSACStyle(nn.Module):
    """
    SAC 风格的 Actor，用在 PPO：
    - 隐藏层: ReLU
    - state-dependent mu / log_std + clamp 到 [-20, 2]
    - a = tanh(u) ∈ [-1,1]
    - logπ(a) 使用数值稳定的公式 (同 SAC/OpenAI 实现)
    """
    def __init__(self, obs_dim, action_dim=1, max_action=40.0,
                 hidden_dim=64):
        super().__init__()
        self.max_action = max_action

        # 简单两层 MLP: obs_dim -> hidden_dim -> hidden_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mu_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

        self.LOG_STD_MAX = 2.0
        self.LOG_STD_MIN = -20.0

    def _dist(self, obs):
        """
        根据 obs 输出 Normal 分布 dist, 以及 mu, log_std
        obs: (obs_dim,) 或 (batch, obs_dim)
        """
        h = self.net(obs)
        mu = self.mu_layer(h)
        log_std = self.log_std_layer(h)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        return dist, mu, log_std

    def forward(self, obs, deterministic=False, with_logprob=True):
        """
        返回:
        - a_norm: tanh(u) ∈ [-1,1]
        - u: pre-tanh sample (同维度)
        - logp_pi_a: log π(a_norm) 若 with_logprob=True，否则 None

        注意：在训练中 logp 是对 a_norm 的概率，但用 u 来计算（数值稳定）。
        """
        dist, mu, log_std = self._dist(obs)

        if deterministic:
            u = mu
        else:
            u = dist.rsample()       # reparameterization trick

        a_norm = torch.tanh(u)       # [-1,1]

        if with_logprob:
            # SAC 论文中的数值稳定公式
            # log π(a) = log N(u; mu, std) - log|d tanh(u)/du|
            # = log N(u) - (2*(log(2) - u - softplus(-2u)))
            logp_u = dist.log_prob(u).sum(dim=-1, keepdim=True)
            correction = (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(
                dim=-1, keepdim=True
            )
            logp_pi_a = logp_u - correction
        else:
            logp_pi_a = None

        return a_norm, u, logp_pi_a

    def act_deterministic(self, obs):
        """
        评估用: 用均值 mu 得到 deterministic 的 a_norm 和 u
        obs: (obs_dim,) tensor
        """
        with torch.no_grad():
            dist, mu, log_std = self._dist(obs)
            u = mu
            a_norm = torch.tanh(u)
        return a_norm.squeeze(-1), u.squeeze(-1)


# ==============================
#  Critic network (global)
# ==============================

class CriticNet(nn.Module):
    def __init__(self, obs_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # (batch,)


# ==============================
#  Helper: split obs by stage
# ==============================

def split_obs_for_stage(global_obs, n_stages, per_stage_obs_dim, stage_id):
    """
    global_obs: np.array 或 1D tensor, shape = (n_stages * per_stage_obs_dim,)
    返回当前 stage 的局部 obs
    """
    start = stage_id * per_stage_obs_dim
    end = start + per_stage_obs_dim
    return global_obs[start:end]


# ==============================
#  Evaluation: deterministic policy
# ==============================

def evaluate_policy(
    n_stages,
    lead_time,
    actors,
    obs_normalizer=None,
    max_action=40.0,
    episode_length=100,
    n_eval_episodes=10,
):
    """
    用确定性策略（mean action）评估当前 actors 的表现。
    - actor 输出 a_norm ∈ [-1,1]
    - 与环境交互时用 a_env = (a_norm+1)/2 * max_action
    - 如提供 obs_normalizer，则只做 normalize，不再更新统计量
    返回：平均每 episode 的真实总 reward
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

        if obs_normalizer is not None:
            norm_global_obs = obs_normalizer.normalize(global_obs)
        else:
            norm_global_obs = global_obs

        done = False
        truncated = False
        ep_return = 0.0

        while not (done or truncated):
            actions_env = []

            for sid in range(n_stages):
                obs_sid = split_obs_for_stage(
                    norm_global_obs, n_stages, per_stage_obs_dim, sid
                )
                obs_sid_t = torch.tensor(obs_sid, dtype=torch.float32)

                a_norm_sid, _ = actors[sid].act_deterministic(obs_sid_t)
                a_norm_val = float(a_norm_sid.item())
                a_norm_val = np.clip(a_norm_val, -1.0, 1.0)

                a_env = (a_norm_val + 1.0) / 2.0 * max_action
                actions_env.append(a_env)

            joint_action = np.array(actions_env, dtype=np.float32)
            next_global_obs, reward, done, truncated, info = env.step(joint_action)

            ep_return += float(np.sum(reward))

            if obs_normalizer is not None:
                norm_global_obs = obs_normalizer.normalize(next_global_obs)
            else:
                norm_global_obs = next_global_obs

            global_obs = next_global_obs

        returns.append(ep_return)

    return float(np.mean(returns))


# ==============================
#  Main Training: PPO
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
    lambda_smooth=5e-3,   # 对 a_norm 的平滑正则
    lead_time=2,
    max_action=40.0,
    use_obs_norm=True,
):
    """
    Federated PPO:
    - 每个 stage 一个 actor
    - critic 是 global state 的共享网络, 每轮对其做 FedAvg
    - actor 使用 SAC 风格的 tanh + 稳定 logprob 实现
    """

    # 1. 环境
    env = SerialMultiStageEnv(
        n_stages=n_stages,
        lead_times=[lead_time] * n_stages,
        episode_length=100,
        render_mode=None,
    )

    per_stage_obs_dim = env.per_stage_obs_dim
    global_obs_dim = env.observation_space.shape[0]

    # 1.5 观测归一化器
    if use_obs_norm:
        obs_normalizer = RunningNormalizer(global_obs_dim)
    else:
        obs_normalizer = None

    # 2. 每个 stage 一个 actor
    actors = []
    actor_optimizers = []
    for i in range(n_stages):
        actor = PPOActorSACStyle(obs_dim=per_stage_obs_dim,
                                 action_dim=1,
                                 max_action=max_action,
                                 hidden_dim=64)
        actors.append(actor)
        actor_optimizers.append(optim.Adam(actor.parameters(), lr=3e-4))

    # 3. 全局 critic (FedAvg)
    global_critic = CriticNet(obs_dim=global_obs_dim, hidden_dim=64)
    critic_lr = 3e-4

    for rnd in range(n_rounds):
        print(f"\n===== Federated Round {rnd+1}/{n_rounds} =====")

        local_critic_params_list = []
        round_rewards_all_steps = []

        # 每个 stage 当作一个 federated client
        for stage_id in range(n_stages):

            local_critic = copy.deepcopy(global_critic)
            critic_optimizer = optim.Adam(local_critic.parameters(), lr=critic_lr)

            # 收集该 stage 的轨迹
            obs_buffer = []
            u_buffer = []           # pre-tanh u
            logprob_buffer = []
            reward_buffer = []
            value_buffer = []
            done_buffer = []

            for ep in range(episodes_per_round):
                global_obs, info = env.reset()

                if obs_normalizer is not None:
                    obs_normalizer.update(global_obs)
                    norm_global_obs = obs_normalizer.normalize(global_obs)
                else:
                    norm_global_obs = global_obs

                done = False
                truncated = False

                while not (done or truncated):
                    actions_env = []
                    log_probs_tmp = []
                    u_tmp = []

                    for sid in range(n_stages):
                        obs_sid = split_obs_for_stage(
                            norm_global_obs, n_stages, per_stage_obs_dim, sid
                        )
                        obs_sid_t = torch.tensor(obs_sid, dtype=torch.float32)

                        a_norm_sid, u_sid, logp_sid = actors[sid](
                            obs_sid_t,
                            deterministic=False,
                            with_logprob=True,
                        )
                        a_norm_val = float(a_norm_sid.detach().numpy()[0])
                        a_norm_val = np.clip(a_norm_val, -1.0, 1.0)

                        a_env = (a_norm_val + 1.0) / 2.0 * max_action

                        actions_env.append(a_env)
                        u_tmp.append(u_sid)
                        log_probs_tmp.append(logp_sid)

                    joint_action = np.array(actions_env, dtype=np.float32)
                    next_global_obs, reward, done, truncated, info = env.step(joint_action)

                    global_reward = np.sum(reward)

                    # critic 使用 global obs（可选归一化）
                    if obs_normalizer is not None:
                        norm_global_obs_tensor = torch.tensor(
                            norm_global_obs, dtype=torch.float32
                        )
                    else:
                        norm_global_obs_tensor = torch.tensor(
                            norm_global_obs, dtype=torch.float32
                        )

                    value = local_critic(norm_global_obs_tensor)

                    # 当前 stage 的局部观测（同样使用 norm_global_obs 切片）
                    local_obs = split_obs_for_stage(
                        norm_global_obs, n_stages, per_stage_obs_dim, stage_id
                    )
                    local_obs_t = torch.tensor(local_obs, dtype=torch.float32)

                    # 缓存当前 stage 的轨迹
                    obs_buffer.append(local_obs_t)
                    u_buffer.append(u_tmp[stage_id].detach())             # pre-tanh u
                    logprob_buffer.append(log_probs_tmp[stage_id])        # log π_old(a)
                    reward_buffer.append(torch.tensor(global_reward, dtype=torch.float32))
                    value_buffer.append(value)
                    done_buffer.append(done or truncated)

                    round_rewards_all_steps.append(global_reward)

                    # 更新 obs
                    if obs_normalizer is not None:
                        obs_normalizer.update(next_global_obs)
                        norm_next_global_obs = obs_normalizer.normalize(next_global_obs)
                    else:
                        norm_next_global_obs = next_global_obs

                    global_obs = next_global_obs
                    norm_global_obs = norm_next_global_obs

            # ============ 3.3 PPO UPDATE for this stage ============

            obs_tensor = torch.stack(obs_buffer)               # (T, obs_dim_stage)
            u_tensor = torch.stack(u_buffer)                   # (T, 1) pre-tanh
            old_logprobs_tensor = torch.stack(logprob_buffer).detach().squeeze(-1)  # (T,)
            rewards_tensor = torch.stack(reward_buffer)        # (T,)
            values_tensor = torch.stack(value_buffer)          # (T,)
            dones_tensor = torch.tensor(done_buffer, dtype=torch.float32)  # (T,)

            T_steps = rewards_tensor.shape[0]

            # ---------- GAE ----------
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

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            returns = returns.detach()

            # 预计算 a_norm (仅用于 smooth penalty)
            a_norm_tensor = torch.tanh(u_tensor)  # (T,1)

            # ---------- PPO Optimization ----------
            eps = 1e-6

            for _ in range(ppo_epochs):
                # 当前 actor 的分布
                dist, mu, log_std = actors[stage_id]._dist(obs_tensor)
                # 用旧的 u_tensor 计算新的 logπ_new(a)
                logp_u_new = dist.log_prob(u_tensor).sum(dim=-1, keepdim=True)
                correction_new = (2 * (np.log(2) - u_tensor - F.softplus(-2 * u_tensor))).sum(
                    dim=-1, keepdim=True
                )
                logp_new = (logp_u_new - correction_new).squeeze(-1)   # (T,)

                entropy = dist.entropy().sum(-1).mean()

                ratio = torch.exp(logp_new - old_logprobs_tensor)      # (T,)

                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - ppo_clip, 1 + ppo_clip) * advantages

                # 平滑正则：惩罚 |a_norm|^2，抑制极端动作倾向
                smooth_penalty = (a_norm_tensor ** 2).mean()

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

            local_critic_params_list.append(copy.deepcopy(local_critic.state_dict()))

        # 3.5 FedAvg 聚合 critic
        print("Server: FedAvg critic parameters.")
        new_state_dict = copy.deepcopy(local_critic_params_list[0])
        for key in new_state_dict.keys():
            for i in range(1, len(local_critic_params_list)):
                new_state_dict[key] += local_critic_params_list[i][key]
            new_state_dict[key] /= float(len(local_critic_params_list))
        global_critic.load_state_dict(new_state_dict)

        # 训练阶段的 step-avg reward（带探索噪声）
        if len(round_rewards_all_steps) > 0:
            avg_reward = np.mean(round_rewards_all_steps)
        else:
            avg_reward = 0.0
        print(f"Round {rnd+1}: approx avg step reward (stochastic, training) = {avg_reward:.2f}")

        # 每隔若干轮做一次 deterministic evaluation
        if (rnd + 1) % 10 == 0:
            eval_return = evaluate_policy(
                n_stages=n_stages,
                lead_time=lead_time,
                actors=actors,
                obs_normalizer=obs_normalizer if use_obs_norm else None,
                max_action=max_action,
                episode_length=100,
                n_eval_episodes=10,
            )
            print(f"[Eval deterministic] Round {rnd+1}: avg episode return = {eval_return:.2f}")


if __name__ == "__main__":
    # 先从 n_stages=1 的简单情况开始
    train_federated_actor_critic(
        n_stages=1,
        n_rounds=500,
        episodes_per_round=100,
        lead_time=2,
        max_action=40.0,
        use_obs_norm=True,   # 如果想完全回到“无标准化版本”，这里改成 False 即可
    )

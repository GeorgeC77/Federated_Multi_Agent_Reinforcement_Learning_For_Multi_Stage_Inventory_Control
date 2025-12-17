import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

from envs.serial_multi_stage import SerialMultiStageEnv


# ================= Replay Buffer =================

class ReplayBuffer:
    def __init__(self, capacity=200000):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        # obs, next_obs: torch.Tensor 1D
        # action: torch.Tensor 1D (shape [1])
        self.buffer.append((
            obs.detach().clone(),
            action.detach().clone(),
            float(reward),
            next_obs.detach().clone(),
            float(done),
        ))

    def sample(self, batch_size, device):
        batch = random.sample(self.buffer, batch_size)
        obs, act, rew, next_obs, done = zip(*batch)
        obs = torch.stack(obs).to(device)               # (B, obs_dim)
        act = torch.stack(act).to(device)               # (B, 1)
        rew = torch.tensor(rew, dtype=torch.float32, device=device)     # (B,)
        next_obs = torch.stack(next_obs).to(device)     # (B, obs_dim)
        done = torch.tensor(done, dtype=torch.float32, device=device)   # (B,)
        return obs, act, rew, next_obs, done

    def __len__(self):
        return len(self.buffer)


# ================= SAC Networks =================

class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim=1, max_action=40.0):
        super().__init__()
        self.max_action = max_action

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.mean = nn.Linear(128, action_dim)
        self.log_std = nn.Linear(128, action_dim)

    def forward(self, obs):
        h = self.net(obs)
        mean = self.mean(h)
        log_std = torch.clamp(self.log_std(h), -5, 2)
        std = torch.exp(log_std)
        return mean, std

    def sample(self, obs):
        """
        obs: (B, obs_dim) or (obs_dim,)
        返回:
          action_env: 映射到 [0, max_action] 的动作，送入 env
          log_prob: 对 raw→tanh 后的 corrected log_prob
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)  # (1, obs_dim)

        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)

        raw = dist.rsample()                   # (B, 1)
        squashed = torch.tanh(raw)             # (B, 1) in [-1,1]

        # 映射到 [0, max_action]
        action = (squashed + 1.0) / 2.0 * self.max_action

        # log_prob 修正
        log_prob = dist.log_prob(raw) - torch.log(1 - squashed.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1)            # (B,)

        return action, log_prob


class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, obs, action):
        """
        obs: (B, obs_dim)
        action: (B, 1)
        """
        x = torch.cat([obs, action], dim=-1)
        return self.net(x).squeeze(-1)  # (B,)


# ================= 训练主函数 =================

def train_sac_single_stage(
    n_rounds=500,
    episodes_per_round=50,
    gamma=0.995,
    tau=0.005,
    buffer_capacity=200000,
    batch_size=256,
    updates_per_step=1,
    max_action=40.0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 环境：先固定 n_stages=1
    env = SerialMultiStageEnv(
        n_stages=1,
        lead_times=[2],
        episode_length=100,
    )

    obs_sample, info = env.reset()
    obs_dim = env.observation_space.shape[0]  # 对 n_stages=1，这就是 per_stage_obs_dim
    action_dim = 1

    # SAC 网络
    policy = GaussianPolicy(obs_dim, action_dim, max_action=max_action).to(device)
    q1 = QNetwork(obs_dim, action_dim).to(device)
    q2 = QNetwork(obs_dim, action_dim).to(device)
    q1_target = QNetwork(obs_dim, action_dim).to(device)
    q2_target = QNetwork(obs_dim, action_dim).to(device)

    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())

    policy_opt = optim.Adam(policy.parameters(), lr=3e-4)
    q1_opt = optim.Adam(q1.parameters(), lr=3e-4)
    q2_opt = optim.Adam(q2.parameters(), lr=3e-4)

    # 自动温度 α
    log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
    alpha_opt = optim.Adam([log_alpha], lr=1e-3)
    target_entropy = -float(action_dim)   # 常用设置

    replay = ReplayBuffer(capacity=buffer_capacity)

    global_step = 0

    for rnd in range(n_rounds):
        total_reward_round = 0.0

        for ep in range(episodes_per_round):
            obs, info = env.reset()
            done = False
            truncated = False

            while not (done or truncated):
                global_step += 1

                obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

                with torch.no_grad():
                    action_env, _ = policy.sample(obs_t)
                action_scalar = action_env.squeeze().cpu().numpy().item()
                # env 期望动作 array，这里只有一个 stage
                joint_action = np.array([action_scalar], dtype=np.float32)

                next_obs, reward, done, truncated, info = env.step(joint_action)
                # global reward = sum(reward)，但这里只有一个 stage
                r = float(np.sum(reward))
                total_reward_round += r

                next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)

                replay.push(obs_t, torch.tensor([action_scalar], dtype=torch.float32, device=device),
                            r, next_obs_t, float(done or truncated))

                obs = next_obs

                # ------- SAC 更新（off-policy，多次更新） -------
                if len(replay) >= batch_size:
                    for _ in range(updates_per_step):
                        obs_b, act_b, rew_b, next_obs_b, done_b = replay.sample(batch_size, device)

                        # 当前 α
                        alpha = log_alpha.exp()

                        # --- 1. 更新 Q 网络 ---
                        with torch.no_grad():
                            # 下一个动作 + log π
                            next_action, next_log_prob = policy.sample(next_obs_b)
                            # 目标 Q 值
                            q1_next = q1_target(next_obs_b, next_action)
                            q2_next = q2_target(next_obs_b, next_action)
                            q_next = torch.min(q1_next, q2_next) - alpha * next_log_prob
                            q_target = rew_b + gamma * (1 - done_b) * q_next

                        q1_pred = q1(obs_b, act_b)
                        q2_pred = q2(obs_b, act_b)

                        q1_loss = ((q1_pred - q_target) ** 2).mean()
                        q2_loss = ((q2_pred - q_target) ** 2).mean()

                        q1_opt.zero_grad()
                        q1_loss.backward()
                        q1_opt.step()

                        q2_opt.zero_grad()
                        q2_loss.backward()
                        q2_opt.step()

                        # --- 2. 更新 policy ---
                        new_action, log_pi = policy.sample(obs_b)
                        q1_new = q1(obs_b, new_action)
                        q2_new = q2(obs_b, new_action)
                        q_new = torch.min(q1_new, q2_new)

                        policy_loss = (alpha * log_pi - q_new).mean()

                        policy_opt.zero_grad()
                        policy_loss.backward()
                        policy_opt.step()

                        # --- 3. 更新 α（自动熵） ---
                        alpha_loss = -(log_alpha * (log_pi.detach() + target_entropy)).mean()
                        alpha_opt.zero_grad()
                        alpha_loss.backward()
                        alpha_opt.step()

                        # --- 4. 软更新 target Q ---
                        with torch.no_grad():
                            for p, tp in zip(q1.parameters(), q1_target.parameters()):
                                tp.data.mul_(1 - tau)
                                tp.data.add_(tau * p.data)

                            for p, tp in zip(q2.parameters(), q2_target.parameters()):
                                tp.data.mul_(1 - tau)
                                tp.data.add_(tau * p.data)

        avg_reward = total_reward_round / episodes_per_round
        print(f"Round {rnd+1}/{n_rounds} | Avg Episode Reward = {avg_reward:.2f}, Buffer size = {len(replay)}")


if __name__ == "__main__":
    train_sac_single_stage()

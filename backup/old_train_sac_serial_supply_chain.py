import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

from envs.serial_multi_stage import SerialMultiStageEnv


# =====================================================================
# SAC NETWORKS
# =====================================================================

class SACActor(nn.Module):
    def __init__(self, obs_dim, action_dim=1, max_action=40):
        super().__init__()
        self.max_action = max_action

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU()
        )

        self.mean = nn.Linear(64, action_dim)
        self.log_std = nn.Linear(64, action_dim)

    def set_action_limit(self, limit):
        self.max_action = limit

    def sample(self, obs):
        h = self.net(obs)
        mean = self.mean(h)
        log_std = torch.clamp(self.log_std(h), -5, 2)
        std = torch.exp(log_std)

        dist = torch.distributions.Normal(mean, std)
        raw = dist.rsample()
        squashed = torch.tanh(raw)
        action = (squashed + 1) / 2 * self.max_action

        # Correct log prob
        log_prob = dist.log_prob(raw) - torch.log(1 - squashed.pow(2) + 1e-6)
        return action, log_prob.sum(-1)


class SACCritic(nn.Module):
    def __init__(self, obs_dim, action_dim=1):
        super().__init__()

        # Double Q heads
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


# =====================================================================
# Replay Buffer
# =====================================================================

class ReplayBuffer:
    def __init__(self, cap=150_000):
        self.buf = deque(maxlen=cap)

    def push(self, *row):
        self.buf.append(row)

    def sample(self, bs):
        batch = random.sample(self.buf, bs)
        return map(np.stack, zip(*batch))

    def __len__(self):
        return len(self.buf)


# =====================================================================
# Utility
# =====================================================================

def extract_local_obs(global_obs, stage, per_dim):
    start = stage * per_dim
    return global_obs[start:start + per_dim]


# =====================================================================
# MAIN TRAIN FUNCTION
# =====================================================================

def train_federated_sac(
        n_stages=3,
        total_rounds=300,
        episodes_per_round=5,
        gamma=0.99,
        alpha=0.2,
        tau=0.01,
        batch_size=256
):

    env = SerialMultiStageEnv(n_stages=n_stages, lead_times=[2]*n_stages, episode_length=100)
    per_dim = env.per_stage_obs_dim
    obs_dim = env.observation_space.shape[0]

    actors = [SACActor(per_dim) for _ in range(n_stages)]
    actor_opt = [optim.Adam(a.parameters(), lr=3e-4) for a in actors]

    global_critic = SACCritic(obs_dim)
    target_critic = copy.deepcopy(global_critic)

    critic_opt = optim.Adam(global_critic.parameters(), lr=3e-4)

    buffer = ReplayBuffer()
    history = []
    warmup = 3000  # replay size before training actor

    print("\n🔥 Federated SAC Training Started (Stable Version)\n")

    for rnd in range(total_rounds):

        # Smoothly increase allowed action range (curriculum)
        # max_action_limit = min(40, 5 + rnd * 0.2)
        max_action_limit = 40
        for a in actors:
            a.set_action_limit(max_action_limit)

        critic_snapshots = []

        for stage in range(n_stages):

            for _ in range(episodes_per_round):
                obs, _ = env.reset()
                done, trunc = False, False
                ep_reward = 0

                while not (done or trunc):

                    actions = []
                    for s in range(n_stages):
                        local_o = extract_local_obs(obs, s, per_dim)
                        obs_t = torch.tensor(local_o, dtype=torch.float32)
                        a, _ = actors[s].sample(obs_t)
                        actions.append(a.item())

                    next_obs, reward, done, trunc, info = env.step(np.array(actions))

                    # ========= REWARD NORMALIZATION =========
                    reward = reward / 1000
                    # reward = np.clip(reward, -50, 5)

                    total_reward = np.sum(reward)
                    ep_reward += total_reward

                    buffer.push(obs, np.array(actions), total_reward, next_obs, float(done))
                    obs = next_obs

                history.append(ep_reward)

            # Start updating only when enough experience
            if len(buffer) < warmup:
                continue

            # ========= UPDATE (SAC) =========
            obs_b, action_b, rew_b, next_b, done_b = buffer.sample(batch_size)

            obs_t = torch.tensor(obs_b, dtype=torch.float32)
            act_t = torch.tensor(action_b[:, stage:stage+1], dtype=torch.float32)
            rew_t = torch.tensor(rew_b, dtype=torch.float32)
            next_t = torch.tensor(next_b, dtype=torch.float32)
            done_t = torch.tensor(done_b, dtype=torch.float32)

            # ---- Target Q ----
            with torch.no_grad():
                next_local_batch = torch.tensor(
                    np.vstack([extract_local_obs(x, stage, per_dim) for x in next_b]),
                    dtype=torch.float32
                )

                next_action, next_logp = actors[stage].sample(next_local_batch)
                q1_next, q2_next = target_critic(next_t, next_action)

                q_target = torch.min(q1_next, q2_next) - alpha * next_logp
                target = rew_t + gamma * (1 - done_t) * q_target

            # ---- Critic Update ----
            q1, q2 = global_critic(obs_t, act_t)
            critic_loss = ((q1 - target)**2 + (q2 - target)**2).mean()

            critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(global_critic.parameters(), 5.0)
            critic_opt.step()

            # ✅ Soft update target_critic（关键补丁）
            with torch.no_grad():
                for p, tp in zip(global_critic.parameters(), target_critic.parameters()):
                    tp.data.mul_(1 - tau)
                    tp.data.add_(tau * p.data)

            # ---- Actor Update ----
            local_obs_batch = torch.tensor(
                np.vstack([extract_local_obs(o, stage, per_dim) for o in obs_b]),
                dtype=torch.float32
            )
            new_action, logp = actors[stage].sample(local_obs_batch)
            q1_pi, q2_pi = global_critic(obs_t, new_action)
            q_pi = torch.min(q1_pi, q2_pi)

            actor_loss = (alpha * logp - q_pi).mean()
            actor_opt[stage].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actors[stage].parameters(), 5.0)
            actor_opt[stage].step()

            critic_snapshots.append(copy.deepcopy(global_critic.state_dict()))

        # ===== FedAvg ======
        # ===== FedAvg or Single-Agent Pass =====
        if n_stages > 1 and len(critic_snapshots) > 0:
            avg_state = copy.deepcopy(critic_snapshots[0])
            for k in avg_state.keys():
                for i in range(1, len(critic_snapshots)):
                    avg_state[k] += critic_snapshots[i][k]
                avg_state[k] /= len(critic_snapshots)

            global_critic.load_state_dict(avg_state)
            target_critic.load_state_dict(avg_state)

        elif n_stages == 1:
            # no federation needed
            pass
        else:
            # buffer not warmed up → skip update
            pass

        # ===== Logging =====
        print(f"Round {rnd+1}/{total_rounds} | Mean Reward (last 20): {np.mean(history[-20:]):.2f} | Buffer={len(buffer)} | Action limit={max_action_limit:.1f}")

    print("\n🎉 Training Completed Successfully.\n")


if __name__ == "__main__":
    train_federated_sac(n_stages=1, total_rounds=500, episodes_per_round=100)

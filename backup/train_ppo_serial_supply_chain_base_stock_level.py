import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from envs.serial_multi_stage import SerialMultiStageEnv


# ================================
# Actor / Critic Network
# ================================

class ActorNet(nn.Module):
    """
    Instead of predicting an absolute order,
    the actor predicts a Base-Stock Level S in [0, max_S].
    """
    def __init__(self, obs_dim, action_dim=1, max_S=100.0):
        super().__init__()
        self.max_S = max_S

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )

        self.mean_head = nn.Linear(64, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * 0.3)

    def forward(self, obs):
        h = self.net(obs)
        mean = torch.tanh(self.mean_head(h)) * self.max_S
        std = self.log_std.exp()
        return mean, std

    def sample(self, obs):
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)

        raw_action = dist.rsample()
        S = torch.clamp(raw_action, 0, self.max_S)

        # Correct log-prob (no tanh squashing now, simpler)
        log_prob = dist.log_prob(raw_action).sum(-1)

        return S, log_prob


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

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


# Helper: extract per-stage local obs
def split_obs(global_obs, n, dim, idx):
    start, end = idx * dim, (idx + 1) * dim
    return global_obs[start:end]


# ================================
# Federated PPO (Base-Stock Learning)
# ================================

def train_federated_base_stock_ppo(
    n_stages=1,
    n_rounds=500,
    episodes_per_round=50,
    gamma=0.99,
    max_S=100.0
):

    env = SerialMultiStageEnv(
        n_stages=n_stages,
        lead_times=[2] * n_stages,
        episode_length=100,
        render_mode=None
    )

    per_obs_dim = env.per_stage_obs_dim
    global_obs_dim = env.observation_space.shape[0]

    # Create actor per supplier
    actors = []
    actor_optimizers = []

    for _ in range(n_stages):
        actor = ActorNet(obs_dim=per_obs_dim, action_dim=1, max_S=max_S)
        actors.append(actor)
        actor_optimizers.append(optim.Adam(actor.parameters(), lr=3e-4))

    # Shared critic
    global_critic = CriticNet(obs_dim=global_obs_dim)
    critic_lr = 3e-4

    print("\n🚀 Training PPO with Base-Stock Action Representation...\n")

    for rnd in range(n_rounds):
        local_critic_snapshots = []

        # ===== Federated Per-Stage Training ======
        for stage_id in range(n_stages):

            local_critic = copy.deepcopy(global_critic)
            critic_optimizer = optim.Adam(local_critic.parameters(), lr=critic_lr)

            obs_buf, logp_buf, reward_buf, val_buf = [], [], [], []

            # rollout
            for _ in range(episodes_per_round):

                obs, _ = env.reset()
                done, truncated = False, False

                while not (done or truncated):

                    joint_actions = []
                    log_probs = []

                    for sid in range(n_stages):
                        local_obs = split_obs(obs, n_stages, per_obs_dim, sid)
                        local_obs_t = torch.tensor(local_obs, dtype=torch.float32)

                        S_pred, logp = actors[sid].sample(local_obs_t)

                        # Convert predicted S into actual order quantity
                        current_inventory = obs[sid * per_obs_dim] - obs[sid * per_obs_dim + 1] + np.sum(obs[sid * per_obs_dim + env.lead_times[sid] : sid * per_obs_dim + env.lead_times[sid] + 1])  # assume first entry = on-hand inventory
                        order = max(0, S_pred.item() - current_inventory)

                        joint_actions.append(order)
                        log_probs.append(logp)

                    next_obs, reward, done, truncated, _ = env.step(np.array(joint_actions))
                    total_reward = np.sum(reward)

                    obs_buf.append(torch.tensor(split_obs(obs, n_stages, per_obs_dim, stage_id), dtype=torch.float32))
                    logp_buf.append(log_probs[stage_id])
                    reward_buf.append(torch.tensor(total_reward, dtype=torch.float32))

                    value = local_critic(torch.tensor(obs, dtype=torch.float32))
                    val_buf.append(value)

                    obs = next_obs

            # ======= Compute returns + advantage =======
            rewards = torch.stack(reward_buf)
            values = torch.stack(val_buf)

            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + gamma * G
                returns.append(G)

            returns.reverse()
            returns = torch.stack(returns).detach()

            adv = returns - values.detach()
            adv = (adv - adv.mean()) / (adv.std() + 1e-6)

            log_probs = torch.stack(logp_buf)

            # ===== PPO Loss (simplified vanilla) =====
            entropy = std_entropy(actors[stage_id], obs_buf)

            actor_loss = -(log_probs * adv).mean() - 0.01 * entropy

            actor_optimizers[stage_id].zero_grad()
            actor_loss.backward()
            actor_optimizers[stage_id].step()

            critic_loss = ((values - returns) ** 2).mean()
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            local_critic_snapshots.append(copy.deepcopy(local_critic.state_dict()))

        # ===== FedAVG =====
        new_state = copy.deepcopy(local_critic_snapshots[0])
        for key in new_state.keys():
            for i in range(1, len(local_critic_snapshots)):
                new_state[key] += local_critic_snapshots[i][key]
            new_state[key] /= len(local_critic_snapshots)

        global_critic.load_state_dict(new_state)

        # Logging
        print(f"Round {rnd+1}/{n_rounds} | Avg Reward: {float(rewards.mean()):.2f}")

    print("\n🎯 Training Complete. Agent now learns base-stock level instead of direct orders.\n")


def std_entropy(actor, obs_list):
    obs = torch.stack(obs_list)
    mean, std = actor.forward(obs)
    return torch.distributions.Normal(mean, std).entropy().mean()



if __name__ == "__main__":
    train_federated_base_stock_ppo(n_stages=1, n_rounds=500, episodes_per_round=100)

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from envs.serial_multi_stage import SerialMultiStageEnv
from utils.utils import RunningNormalizer     # using your existing utility


# ---------------- Actor / Critic ----------------

class ActorNet(nn.Module):
    def __init__(self, obs_dim, action_dim=1, max_action=40.0):
        super().__init__()
        self.max_action = max_action

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )

        self.mean_head = nn.Linear(64, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * 0.5)

    def forward(self, x):
        h = self.net(x)
        mean = self.mean_head(h)
        mean = torch.tanh(mean) * self.max_action
        std = self.log_std.exp()
        return mean, std

    def sample(self, obs):
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.rsample()

        squashed = torch.tanh(raw_action)
        action = (squashed + 1) / 2 * self.max_action

        log_prob = dist.log_prob(raw_action) - torch.log(1 - squashed.pow(2) + 1e-6)
        return action, log_prob.sum(-1)


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
        return self.net(x).squeeze(-1)


# ---------------- Helper ----------------

def split_obs_for_stage(global_obs_normed, n_stages, per_stage_dim, stage_id):
    start = stage_id * per_stage_dim
    end = start + per_stage_dim
    return global_obs_normed[start:end]


# ---------------- Main PPO Federated Training ----------------

def train_federated_actor_critic(
        n_stages=2,
        n_rounds=100,
        episodes_per_round=10,
        gamma=0.99,
):
    env = SerialMultiStageEnv(
        n_stages=n_stages,
        lead_times=[2] * n_stages,
        episode_length=100,
    )

    per_stage_dim = env.per_stage_obs_dim
    global_obs_dim = env.observation_space.shape[0]

    # ---- normalization storage ----
    global_normalizer = RunningNormalizer(global_obs_dim)

    # ---- actors ----
    actors = []
    actor_optimizers = []
    for _ in range(n_stages):
        actor = ActorNet(per_stage_dim)
        actors.append(actor)
        actor_optimizers.append(optim.Adam(actor.parameters(), lr=3e-4))

    # ---- global critic ----
    global_critic = CriticNet(global_obs_dim)
    critic_lr = 3e-4

    print("\n🔧 Training PPO with Normalized Observations...\n")

    for rnd in range(n_rounds):
        local_critic_params = []
        local_norm_stats = []

        for stage_id in range(n_stages):

            local_critic = copy.deepcopy(global_critic)
            critic_optimizer = optim.Adam(local_critic.parameters(), lr=critic_lr)

            obs_buf, logp_buf, reward_buf, value_buf = [], [], [], []

            # ---- rollout ----
            for _ in range(episodes_per_round):
                global_obs, _ = env.reset()
                done = truncated = False

                while not (done or truncated):

                    # ---- update normalizer ----
                    global_normalizer.update(global_obs)

                    # ---- normalize ----
                    normed_obs = global_normalizer.normalize(global_obs)

                    # ---- extract local observation ----
                    local_obs = split_obs_for_stage(normed_obs, n_stages, per_stage_dim, stage_id)
                    local_obs_t = torch.tensor(local_obs, dtype=torch.float32)

                    # ---- sample joint action ----
                    actions = []
                    logpi_stage = None

                    for sid in range(n_stages):
                        sid_obs = split_obs_for_stage(normed_obs, n_stages, per_stage_dim, sid)
                        sid_obs_t = torch.tensor(sid_obs, dtype=torch.float32)
                        a, lp = actors[sid].sample(sid_obs_t)

                        if sid == stage_id:
                            logpi_stage = lp

                        actions.append(a.item())

                    next_obs, reward, done, truncated, _ = env.step(np.array(actions, dtype=np.float32))

                    # critic input also normalized
                    value = local_critic(torch.tensor(normed_obs, dtype=torch.float32))

                    obs_buf.append(normed_obs)
                    logp_buf.append(logpi_stage)
                    reward_buf.append(sum(reward))  # global reward
                    value_buf.append(value)

                    global_obs = next_obs

            # ---- compute advantages ----
            rewards = torch.tensor(reward_buf, dtype=torch.float32)
            values = torch.stack(value_buf)

            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + gamma * G
                returns.append(G)
            returns.reverse()
            returns = torch.tensor(returns, dtype=torch.float32)

            adv = returns - values.detach()
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            # ---- update actor ----
            policy_loss = -(torch.stack(logp_buf) * adv).mean()
            actor_optimizers[stage_id].zero_grad()
            policy_loss.backward()
            actor_optimizers[stage_id].step()

            # ---- update critic ----
            critic_loss = ((values - returns) ** 2).mean()
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # upload critic parameters + normalization stats
            local_critic_params.append(copy.deepcopy(local_critic.state_dict()))
            local_norm_stats.append(global_normalizer.get_stats())

        # ---------------- FedAvg Aggregation ----------------

        new_state = copy.deepcopy(local_critic_params[0])
        for key in new_state:
            for i in range(1, n_stages):
                new_state[key] += local_critic_params[i][key]
            new_state[key] /= n_stages
        global_critic.load_state_dict(new_state)

        # ---- FedAvg for normalizer ----
        mean = sum([s["mean"] for s in local_norm_stats]) / n_stages
        var = sum([s["var"] for s in local_norm_stats]) / n_stages
        global_normalizer.set_stats(mean, var)

        print(f"Round {rnd+1}/{n_rounds} | Mean reward ≈ {rewards.mean().item():.2f}")


if __name__ == "__main__":
    train_federated_actor_critic(n_stages=1, n_rounds=500, episodes_per_round=100)

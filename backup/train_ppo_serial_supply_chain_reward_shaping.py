import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from envs.serial_multi_stage import SerialMultiStageEnv

from utils.utils import RunningNormalizer


# --------- Actor / Critic network ---------

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

        # Instead of 0 → give more exploration initially
        self.log_std = nn.Parameter(torch.ones(action_dim) * 0.5)

    def forward(self, x):
        h = self.net(x)
        mean = self.mean_head(h)

        # Bound mean to a reasonable range (optional stabilizer)
        mean = torch.tanh(mean) * self.max_action

        std = self.log_std.exp()
        return mean, std

    def sample(self, obs):
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)

        # raw Gaussian sample
        raw_action = dist.rsample()  # rsample → reparameterization trick

        # Tanh squashing → enforce bounded action
        squashed = torch.tanh(raw_action)

        # map [-1,1] → [0, max_action]
        action = (squashed + 1) / 2 * self.max_action

        # log_prob correction for tanh squashing (important for PPO)
        log_prob = dist.log_prob(raw_action) - torch.log(1 - squashed.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1)

        return action, log_prob


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


# --------- Extract a local observation from a specific stage within the global observation ---------

def split_obs_for_stage(global_obs, n_stages, per_stage_obs_dim, stage_id):
    """
    global_obs: np.array, shape = (n_stages * per_stage_obs_dim,)
    Return the local obs vector for stage_id
    """
    start = stage_id * per_stage_obs_dim
    end = start + per_stage_obs_dim
    return global_obs[start:end]


# --------- Main Training Process (with reward shaping) ---------

def train_federated_actor_critic(
    n_stages=2,
    n_rounds=100,
    episodes_per_round=10,
    gamma=0.99,
    beta=0.1,        # reward shaping weight
    target_stock=20, # shaping target inventory position (for now fixed; later可改成动态)
):
    # 1. Create environment (single serial supply chain)
    env = SerialMultiStageEnv(
        n_stages=n_stages,
        lead_times=[2] * n_stages,
        episode_length=100,
        render_mode=None,
    )

    per_stage_obs_dim = env.per_stage_obs_dim
    global_obs_dim = env.observation_space.shape[0]

    # 2. Create an actor for each stage
    actors = []
    actor_optimizers = []
    for i in range(n_stages):
        actor = ActorNet(obs_dim=per_stage_obs_dim, action_dim=1)
        actors.append(actor)
        actor_optimizers.append(optim.Adam(actor.parameters(), lr=3e-4))

    # 3. Create a “global critic” whose parameters will be updated via FedAvg.
    global_critic = CriticNet(obs_dim=global_obs_dim)

    # Note: To simulate FL, we clone n_stages local_critic instances per round.
    # Each stage updates its local_critic locally using its own trajectory, then aggregates via FedAvg.
    critic_lr = 3e-4

    print("🚀 Training federated actor-critic with reward shaping...")

    for rnd in range(n_rounds):
        print(f"\n===== Federated Round {rnd+1}/{n_rounds} =====")

        # Collect all local critic parameters from each stage for subsequent FedAvg processing.
        local_critic_params_list = []

        # Each stage is treated as a separate FL client.
        for stage_id in range(n_stages):
            # 3.1 Clone the current global critic as a local critic
            local_critic = copy.deepcopy(global_critic)
            critic_optimizer = optim.Adam(local_critic.parameters(), lr=critic_lr)

            # 3.2 Collect trajectories for this stage across multiple episodes (global environment interaction, but only update actors within this stage)
            obs_buffer = []
            action_buffer = []
            logprob_buffer = []
            reward_buffer = []
            value_buffer = []

            for ep in range(episodes_per_round):
                global_obs, info = env.reset()
                done = False
                truncated = False

                while not (done or truncated):
                    # Extract the local observations for this stage from global_obs
                    local_obs = split_obs_for_stage(
                        global_obs, n_stages, per_stage_obs_dim, stage_id
                    )
                    local_obs_t = torch.tensor(local_obs, dtype=torch.float32)

                    # All actors select actions (Note: Policies from other stages also participate in the environment)
                    actions = []
                    log_probs = []
                    for sid in range(n_stages):
                        obs_sid = split_obs_for_stage(
                            global_obs, n_stages, per_stage_obs_dim, sid
                        )
                        obs_sid_t = torch.tensor(obs_sid, dtype=torch.float32)
                        a_sid, logp_sid = actors[sid].sample(obs_sid_t)
                        actions.append(a_sid.detach().numpy()[0])
                        log_probs.append(logp_sid)  # only for stage_id

                    joint_action = np.array(actions, dtype=np.float32)

                    next_global_obs, reward, done, truncated, info = env.step(joint_action)

                    # Using global reward
                    global_reward = np.sum(reward)

                    # critic uses global state
                    value = local_critic(torch.tensor(global_obs, dtype=torch.float32))

                    # === REWARD SHAPING 部分（唯一核心改动）===
                    # local_obs[0] = inventory of this stage
                    inventory_position = float(local_obs[0])
                    shaping_penalty = abs(inventory_position - target_stock)
                    shaped_reward = global_reward - beta * shaping_penalty
                    # ========================================

                    # Store traces only for the current stage
                    obs_buffer.append(local_obs_t)
                    action_buffer.append(torch.tensor([actions[stage_id]], dtype=torch.float32))
                    logprob_buffer.append(log_probs[stage_id])

                    # 原本是 global_reward，这里换成 shaped_reward
                    reward_buffer.append(torch.tensor(shaped_reward, dtype=torch.float32))
                    value_buffer.append(value)

                    global_obs = next_global_obs

            # 3.3 Calculate return/advantage using the buffer, update the actor and local critic for the current stage

            rewards = torch.stack(reward_buffer)
            values = torch.stack(value_buffer)

            returns = []
            G = 0.0
            for r in reversed(rewards):
                G = r + gamma * G
                returns.append(G)
            returns.reverse()
            returns = torch.stack(returns).detach()

            # ---------- advantage ---- normalize ----------
            advantages = returns - values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # ---------- policy entropy for exploration ----------
            logprobs = torch.stack(logprob_buffer)
            # compute entropy from stored mean/std（这里简单用第一条obs估计一个entropy）
            with torch.no_grad():
                mean, std = actors[stage_id].forward(obs_buffer[0])
            entropy = torch.distributions.Normal(mean, std).entropy().mean()

            # ---------- actor loss ----------
            actor_loss = -(logprobs * advantages).mean() - 0.01 * entropy

            actor_optimizers[stage_id].zero_grad()
            actor_loss.backward()
            actor_optimizers[stage_id].step()

            # critic loss
            critic_loss = ((values - returns) ** 2).mean()
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # 3.4 Add the local critic parameter to the list (simulating upload to the server)
            local_critic_params_list.append(copy.deepcopy(local_critic.state_dict()))

        # 3.5 Server: FedAvg Aggregation Critic Parameter
        print("Server: FedAvg critic parameters.")
        # Simple average
        new_state_dict = copy.deepcopy(local_critic_params_list[0])
        for key in new_state_dict.keys():
            # accumulation
            for i in range(1, len(local_critic_params_list)):
                new_state_dict[key] += local_critic_params_list[i][key]
            new_state_dict[key] /= float(len(local_critic_params_list))
        global_critic.load_state_dict(new_state_dict)

        # (Optional) Print the average shaped reward per round
        avg_reward = rewards.mean().item()
        print(
            f"Round {rnd+1}: approx last-episode avg SHAPED reward "
            f"= {avg_reward:.2f} (beta={beta}, target={target_stock})"
        )


if __name__ == "__main__":
    # 单 stage 验证
    train_federated_actor_critic(n_stages=1, n_rounds=500, episodes_per_round=100)

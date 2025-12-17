# train_frl_ppo_strict.py
# Strict Federated PPO (no data sharing across clients)
# - Each stage is a separate client process holding its own actor/critic/buffer
# - Coordinator holds env and only sends local obs slice + global reward scalar
# - FedAvg aggregates ONLY critic parameters (no trajectories)

import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim
import multiprocessing as mp

from envs.serial_multi_stage import SerialMultiStageEnv
from utils.utils import RunningNormalizer


# ==============================
#  SAC-style Actor for PPO
#  output normalized action a_norm ∈ [-1,1]
#  env action = (a_norm+1)/2 * max_action (absolute order quantity)
# ==============================
class PPOActorSACStyle(nn.Module):
    def __init__(self, obs_dim, action_dim=1, max_action=40.0, hidden_dim=64):
        super().__init__()
        self.max_action = max_action

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
        h = self.net(obs)
        mu = self.mu_layer(h)
        log_std = self.log_std_layer(h)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        return dist, mu, log_std

    def forward(self, obs, deterministic=False, with_logprob=True):
        dist, mu, log_std = self._dist(obs)
        if deterministic:
            u = mu
        else:
            u = dist.rsample()

        a_norm = torch.tanh(u)

        if with_logprob:
            logp_u = dist.log_prob(u).sum(dim=-1, keepdim=True)
            correction = (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(dim=-1, keepdim=True)
            logp_pi_a = logp_u - correction
        else:
            logp_pi_a = None

        return a_norm, u, logp_pi_a

    def act(self, obs_np, deterministic=False):
        """
        obs_np: np.ndarray shape (obs_dim,)
        returns:
          action_env (float), u (tensor shape (1,1)), logp (tensor shape (1,1)), a_norm (tensor shape (1,1))
        """
        obs_t = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0)
        a_norm, u, logp = self.forward(obs_t, deterministic=deterministic, with_logprob=True)
        a_norm_val = float(a_norm.detach().cpu().numpy()[0, 0])
        a_norm_val = np.clip(a_norm_val, -1.0, 1.0)
        action_env = (a_norm_val + 1.0) / 2.0 * self.max_action
        return float(action_env), u.detach(), logp.detach(), a_norm.detach()


# ==============================
#  Local Critic (value function)
#  Strict FL: critic uses LOCAL obs only
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
        return self.net(x).squeeze(-1)


# ==============================
#  Client Worker (one stage)
# ==============================
def client_worker(
    stage_id: int,
    obs_dim: int,
    max_action: float,
    actor_lr: float,
    critic_lr: float,
    gamma: float,
    gae_lambda: float,
    ppo_epochs: int,
    ppo_clip: float,
    entropy_coef: float,
    lambda_smooth: float,
    use_obs_norm: bool,
    conn,                  # Pipe connection
    seed: int = 0,
):
    torch.manual_seed(seed + 1000 * stage_id)
    np.random.seed(seed + 1000 * stage_id)

    actor = PPOActorSACStyle(obs_dim=obs_dim, action_dim=1, max_action=max_action, hidden_dim=64)
    critic = CriticNet(obs_dim=obs_dim, hidden_dim=64)

    actor_opt = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_opt = optim.Adam(critic.parameters(), lr=critic_lr)

    obs_norm = RunningNormalizer(obs_dim) if use_obs_norm else None

    # Local rollout buffers (NEVER leave this process)
    obs_buf = []
    u_buf = []
    logp_buf = []
    val_buf = []
    rew_buf = []
    done_buf = []

    # We store the latest transition slot for reward feedback matching
    # Here we append obs/u/logp/value at act-time; reward/done appended later via feedback.
    pending_steps = 0

    def clear_buffers():
        nonlocal obs_buf, u_buf, logp_buf, val_buf, rew_buf, done_buf, pending_steps
        obs_buf = []
        u_buf = []
        logp_buf = []
        val_buf = []
        rew_buf = []
        done_buf = []
        pending_steps = 0

    def ppo_update():
        nonlocal obs_buf, u_buf, logp_buf, val_buf, rew_buf, done_buf

        if len(obs_buf) == 0:
            return

        obs_t = torch.tensor(np.array(obs_buf, dtype=np.float32), dtype=torch.float32)  # (T, obs_dim)
        u_t = torch.stack(u_buf).squeeze(1)                                             # (T,1)
        old_logp_t = torch.stack(logp_buf).squeeze(1).squeeze(-1).detach()              # (T,)
        values_t = torch.stack(val_buf).detach()                                        # (T,)
        rewards_t = torch.tensor(np.array(rew_buf, dtype=np.float32), dtype=torch.float32)  # (T,)
        dones_t = torch.tensor(np.array(done_buf, dtype=np.float32), dtype=torch.float32)   # (T,)

        T = rewards_t.shape[0]
        advantages = torch.zeros(T, dtype=torch.float32)
        returns = torch.zeros(T, dtype=torch.float32)

        next_value = 0.0
        next_adv = 0.0

        for t in reversed(range(T)):
            if dones_t[t] > 0.5:
                next_value = 0.0
                next_adv = 0.0

            delta = rewards_t[t] + gamma * next_value - values_t[t]
            next_adv = delta + gamma * gae_lambda * next_adv

            advantages[t] = next_adv
            returns[t] = advantages[t] + values_t[t]
            next_value = values_t[t]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = returns.detach()

        # Smooth penalty uses a_norm = tanh(u)
        a_norm_t = torch.tanh(u_t)  # (T,1)

        for _ in range(ppo_epochs):
            dist, mu, log_std = actor._dist(obs_t)
            logp_u_new = dist.log_prob(u_t).sum(dim=-1, keepdim=True)
            correction_new = (2 * (np.log(2) - u_t - F.softplus(-2 * u_t))).sum(dim=-1, keepdim=True)
            logp_new = (logp_u_new - correction_new).squeeze(-1)  # (T,)

            entropy = dist.entropy().sum(-1).mean()
            ratio = torch.exp(logp_new - old_logp_t)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - ppo_clip, 1 + ppo_clip) * advantages

            smooth_penalty = (a_norm_t ** 2).mean()

            actor_loss = -torch.min(surr1, surr2).mean() - entropy_coef * entropy + lambda_smooth * smooth_penalty

            actor_opt.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
            actor_opt.step()

        # Critic update (local)
        v_pred = critic(obs_t).view(-1)
        critic_loss = ((v_pred - returns) ** 2).mean()

        critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
        critic_opt.step()

    clear_buffers()

    while True:
        msg = conn.recv()
        cmd = msg.get("cmd", None)

        if cmd == "close":
            conn.close()
            break

        elif cmd == "sync_critic":
            # Receive global critic params (FedAvg result)
            state_dict = msg["critic_state_dict"]
            critic.load_state_dict(state_dict)
            conn.send({"ok": True})

        elif cmd == "reset_rollout":
            clear_buffers()
            conn.send({"ok": True})

        elif cmd == "act":
            # Receive local obs slice; return action_env only
            obs_local = msg["obs"]  # np.ndarray
            deterministic = bool(msg.get("deterministic", False))

            if obs_norm is not None:
                obs_norm.update(obs_local)
                obs_in = obs_norm.normalize(obs_local)
            else:
                obs_in = obs_local

            # action
            action_env, u, logp, a_norm = actor.act(obs_in, deterministic=deterministic)

            # store local transition parts (NO sharing)
            obs_buf.append(obs_in.copy())
            u_buf.append(u)         # (1,1)
            logp_buf.append(logp)   # (1,1)

            # value estimate from local critic
            with torch.no_grad():
                v = critic(torch.tensor(obs_in, dtype=torch.float32).unsqueeze(0)).squeeze(0)
            val_buf.append(v.detach())

            pending_steps += 1

            conn.send({"action": float(action_env)})

        elif cmd == "feedback":
            # Receive reward signal (scalar) + done flag
            # reward is global scalar (same for all clients) to optimize system-wide
            r = float(msg["reward"])
            done = bool(msg["done"])
            rew_buf.append(r)
            done_buf.append(1.0 if done else 0.0)
            conn.send({"ok": True})

        elif cmd == "update":
            # Perform PPO update locally, then send critic params for FedAvg
            ppo_update()
            # Send critic state dict only (no data)
            conn.send({"critic_state_dict": copy.deepcopy(critic.state_dict())})

        elif cmd == "get_actor_state":
            # for evaluation: share actor params only
            conn.send({"actor_state_dict": copy.deepcopy(actor.state_dict())})

        else:
            conn.send({"error": f"Unknown cmd: {cmd}"})


# ==============================
#  Helper: slice local obs from global obs (coordinator side)
#  NOTE: coordinator can see global obs (env is centralized simulator)
# ==============================
def split_obs_for_stage(global_obs, n_stages, per_stage_obs_dim, stage_id):
    start = stage_id * per_stage_obs_dim
    end = start + per_stage_obs_dim
    return global_obs[start:end]


# ==============================
#  Deterministic evaluation (coordinator side)
# ==============================
@torch.no_grad()
def evaluate_policy_strict(
    env: SerialMultiStageEnv,
    actors_state_dicts,
    n_stages: int,
    per_stage_obs_dim: int,
    max_action: float,
    n_eval_episodes: int = 20,
):
    # Rebuild actors locally for eval only (params are not data)
    actors = []
    for sid in range(n_stages):
        a = PPOActorSACStyle(obs_dim=per_stage_obs_dim, action_dim=1, max_action=max_action, hidden_dim=64)
        a.load_state_dict(actors_state_dicts[sid])
        a.eval()
        actors.append(a)

    ep_returns_global = []
    ep_returns_stage = []

    for ep in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        ep_ret = 0.0
        ep_stage = np.zeros(n_stages, dtype=np.float32)

        while not (done or truncated):
            actions = []
            for sid in range(n_stages):
                obs_i = split_obs_for_stage(obs, n_stages, per_stage_obs_dim, sid)
                obs_t = torch.tensor(obs_i, dtype=torch.float32).unsqueeze(0)
                # deterministic: use mu -> tanh(mu)
                dist, mu, _ = actors[sid]._dist(obs_t)
                a_norm = torch.tanh(mu)[0, 0].item()
                a_norm = float(np.clip(a_norm, -1.0, 1.0))
                a_env = (a_norm + 1.0) / 2.0 * max_action
                actions.append(a_env)

            obs, reward_vec, done, truncated, info = env.step(np.array(actions, dtype=np.float32))
            ep_ret += float(np.sum(reward_vec))
            ep_stage += np.array(reward_vec, dtype=np.float32)

        ep_returns_global.append(ep_ret)
        ep_returns_stage.append(ep_stage.copy())

    return float(np.mean(ep_returns_global)), np.mean(np.stack(ep_returns_stage, axis=0), axis=0)


# ==============================
#  Coordinator: strict FL training loop
# ==============================
def train_strict_federated_ppo(
    n_stages=2,
    lead_times=2,                 # int or list
    n_rounds=200,
    episodes_per_round=10,
    episode_length=100,
    max_action=40.0,
    # PPO params
    gamma=0.99,
    gae_lambda=0.97,
    ppo_epochs=10,
    ppo_clip=0.2,
    entropy_coef=0.01,
    lambda_smooth=5e-3,
    # opt
    actor_lr=3e-4,
    critic_lr=3e-4,
    use_obs_norm=True,
    seed=0,
    eval_every=10,
    eval_episodes=50,
):
    # lead_times normalize
    if isinstance(lead_times, (int, float)):
        lt = [int(lead_times)] * n_stages
    else:
        lt = list(lead_times)
        assert len(lt) == n_stages

    # env lives ONLY in coordinator
    env = SerialMultiStageEnv(
        n_stages=n_stages,
        lead_times=lt,
        episode_length=episode_length,
        max_order=max_action,
        render_mode=None,
        seed=seed,
    )

    per_stage_obs_dim = env.per_stage_obs_dim

    # Start client processes
    ctx = mp.get_context("spawn")  # safer on macOS
    conns = []
    procs = []

    for sid in range(n_stages):
        parent_conn, child_conn = ctx.Pipe()
        p = ctx.Process(
            target=client_worker,
            args=(
                sid, per_stage_obs_dim, max_action,
                actor_lr, critic_lr,
                gamma, gae_lambda,
                ppo_epochs, ppo_clip,
                entropy_coef, lambda_smooth,
                use_obs_norm,
                child_conn,
                seed,
            ),
        )
        p.daemon = True
        p.start()
        conns.append(parent_conn)
        procs.append(p)

    # Initialize global critic state as average of initial critics
    # (Ask each client to "update" with empty buffers would be weird; instead we just fetch their critic by forcing a sync from one)
    # We'll create a fresh critic and broadcast it.
    global_critic = CriticNet(obs_dim=per_stage_obs_dim, hidden_dim=64)
    global_critic_state = global_critic.state_dict()

    # Broadcast initial critic
    for c in conns:
        c.send({"cmd": "sync_critic", "critic_state_dict": global_critic_state})
        c.recv()

    best_eval = -np.inf

    for rnd in range(1, n_rounds + 1):
        print(f"\n===== Strict FL Round {rnd}/{n_rounds} =====")

        # reset local rollouts
        for c in conns:
            c.send({"cmd": "reset_rollout"})
        for c in conns:
            c.recv()

        # rollout for episodes_per_round episodes
        train_step_rewards_global = []
        train_step_rewards_stage = []

        for ep in range(episodes_per_round):
            obs, _ = env.reset()
            done = False
            truncated = False

            while not (done or truncated):
                # send local obs to each client, receive actions
                actions = []
                for sid, c in enumerate(conns):
                    obs_i = split_obs_for_stage(obs, n_stages, per_stage_obs_dim, sid)
                    c.send({"cmd": "act", "obs": obs_i, "deterministic": False})
                for c in conns:
                    actions.append(c.recv()["action"])

                obs, reward_vec, done, truncated, info = env.step(np.array(actions, dtype=np.float32))

                global_r = float(np.sum(reward_vec))

                # feedback: send global scalar reward to all clients (strict, no per-stage sharing needed)
                for c in conns:
                    c.send({"cmd": "feedback", "reward": global_r, "done": bool(done or truncated)})
                for c in conns:
                    c.recv()

                train_step_rewards_global.append(global_r)
                train_step_rewards_stage.append(np.array(reward_vec, dtype=np.float32))

        # Local update on each client and collect critic params for FedAvg
        critic_states = []
        for c in conns:
            c.send({"cmd": "update"})
        for c in conns:
            critic_states.append(c.recv()["critic_state_dict"])

        # FedAvg critic parameters
        new_state = copy.deepcopy(critic_states[0])
        for k in new_state.keys():
            for i in range(1, len(critic_states)):
                new_state[k] += critic_states[i][k]
            new_state[k] /= float(len(critic_states))
        global_critic_state = new_state

        # Broadcast updated global critic
        for c in conns:
            c.send({"cmd": "sync_critic", "critic_state_dict": global_critic_state})
        for c in conns:
            c.recv()

        # Print training reward (stochastic)
        if len(train_step_rewards_global) > 0:
            avg_global = float(np.mean(train_step_rewards_global))
            avg_stage = np.mean(np.stack(train_step_rewards_stage, axis=0), axis=0)
        else:
            avg_global = 0.0
            avg_stage = np.zeros(n_stages, dtype=np.float32)

        print(f"[Train] avg step global reward = {avg_global:.2f}, per-stage = {avg_stage}")

        # Evaluation
        if (rnd % eval_every) == 0:
            # fetch actors
            actor_states = []
            for c in conns:
                c.send({"cmd": "get_actor_state"})
            for c in conns:
                actor_states.append(c.recv()["actor_state_dict"])

            eval_env = SerialMultiStageEnv(
                n_stages=n_stages,
                lead_times=lt,
                episode_length=episode_length,
                max_order=max_action,
                render_mode=None,
                seed=seed + 999,
            )
            eval_global, eval_stage = evaluate_policy_strict(
                env=eval_env,
                actors_state_dicts=actor_states,
                n_stages=n_stages,
                per_stage_obs_dim=per_stage_obs_dim,
                max_action=max_action,
                n_eval_episodes=eval_episodes,
            )
            print(f"[Eval deterministic] mean total return = {eval_global:.2f}, mean per-stage = {eval_stage}")
            if eval_global > best_eval:
                best_eval = eval_global
            print(f"[Best eval] {best_eval:.2f}")

    # close clients
    for c in conns:
        c.send({"cmd": "close"})
    for p in procs:
        p.join(timeout=1.0)


if __name__ == "__main__":
    train_strict_federated_ppo(
        n_stages=2,
        lead_times=[2, 2],
        n_rounds=500,
        episodes_per_round=100,
        episode_length=100,
        max_action=40.0,
        gamma=0.99,
        gae_lambda=0.97,
        ppo_epochs=10,
        ppo_clip=0.2,
        entropy_coef=0.01,
        lambda_smooth=5e-3,
        actor_lr=3e-4,
        critic_lr=3e-4,
        use_obs_norm=True,     # strict: each client normalizes its own local obs
        seed=0,
        eval_every=10,
        eval_episodes=100,
    )

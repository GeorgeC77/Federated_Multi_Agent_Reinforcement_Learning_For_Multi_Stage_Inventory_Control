import numpy as np
from envs.serial_multi_stage import SerialMultiStageEnv

def simulate_base_stock(S, n_steps=20000, seed=0):
    env = SerialMultiStageEnv(
        n_stages=1,
        lead_times=[2],          # 用你训练时的一模一样参数
        episode_length=10**9,    # 足够长，不用管 truncate
        render_mode=None,
        seed=seed,
    )
    obs, info = env.reset()

    total_reward = 0.0
    t = 0
    done = False
    truncated = False

    while t < n_steps:
        # === inventory position: on-hand + pipeline - backlog ===
        inv = env.inventory[0]
        back = env.backlog[0]
        pipe = env.pipeline[0]
        IP = inv + pipe.sum() - back

        order = max(0.0, min(env.max_order, S - IP))
        obs, reward, done, truncated, info = env.step(np.array([order], dtype=np.float32))

        total_reward += reward[0]
        t += 1

        if truncated:          # 保险起见
            obs, info = env.reset()

    return total_reward / n_steps  # 平均每期 reward


if __name__ == "__main__":
    for S in range(35, 60, 1):
        avg_r = simulate_base_stock(S, n_steps=20000, seed=1)
        print(f"S = {S:2d}  →  avg reward ≈ {avg_r:.2f}")

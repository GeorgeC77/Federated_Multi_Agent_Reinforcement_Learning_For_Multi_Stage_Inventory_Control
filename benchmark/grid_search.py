import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

from envs.serial_multi_stage import SerialMultiStageEnv


# -----------------------------
# 你只需要保证这个解析函数与 env 的 obs 对齐
# -----------------------------
def split_obs_for_stage(global_obs: np.ndarray, n_stages: int, per_stage_obs_dim: int, stage_id: int) -> np.ndarray:
    start = stage_id * per_stage_obs_dim
    end = start + per_stage_obs_dim
    return global_obs[start:end]


def extract_inventory_position_from_stage_obs(stage_obs: np.ndarray, lead_time: int) -> float:
    """
    默认假设：
      stage_obs[0] = on_hand
      stage_obs[1] = backlog/backorder (>=0)
      stage_obs[2:2+lead_time] = pipeline
    inventory position = on_hand - backlog + sum(pipeline)

    如果你的 env 定义不同，请改这里即可。
    """
    on_hand = float(stage_obs[0])
    backlog = float(stage_obs[1])
    pipeline = stage_obs[2: 2 + lead_time]
    ip = on_hand - backlog + float(np.sum(pipeline))
    return ip


def echelon_base_stock_action(global_obs: np.ndarray,
                             env: SerialMultiStageEnv,
                             S1: float,
                             S2: float,
                             max_action: float) -> np.ndarray:
    """
    Two-stage echelon base-stock:
      E1 = IP1
      E2 = IP1 + IP2
      q1 = [S1 - E1]^+
      q2 = [S2 - E2]^+
    """
    assert env.n_stages == 2, "This helper is for 2-stage only (extend if needed)."
    per_dim = env.per_stage_obs_dim
    L1, L2 = env.lead_times  # 需要你 env 已按 list[int] 保存

    obs1 = split_obs_for_stage(global_obs, 2, per_dim, 0)
    obs2 = split_obs_for_stage(global_obs, 2, per_dim, 1)

    IP1 = extract_inventory_position_from_stage_obs(obs1, L1)
    IP2 = extract_inventory_position_from_stage_obs(obs2, L2)

    E1 = IP1
    E2 = IP1 + IP2

    q1 = max(0.0, S1 - E1)
    q2 = max(0.0, S2 - E2)

    # 限幅到 env 的 action 上限（如果 env 本身会 clip，也可不做）
    q1 = float(np.clip(q1, 0.0, max_action))
    q2 = float(np.clip(q2, 0.0, max_action))

    return np.array([q1, q2], dtype=np.float32)


# -----------------------------
# 仿真评估一个 (S1, S2)
# -----------------------------
def evaluate_S1S2(env_cfg: Dict[str, Any],
                 S1: float,
                 S2: float,
                 n_episodes: int = 50,
                 max_action: float = 40.0,
                 warmup_periods: int = 50,
                 seed: Optional[int] = None) -> Tuple[float, np.ndarray]:
    """
    返回：
      mean_total_return: 平均 episode 总 reward（只统计 warm-up 之后）
      mean_stage_return: shape (2,) 每个 stage 的平均 episode 总 reward

    关键：
      - 前 warmup_periods 不计 reward（burn-in）
      - 使用同一个 (S1,S2) policy
    """
    env = SerialMultiStageEnv(**env_cfg)

    total_returns = []
    stage_returns = []

    for ep in range(n_episodes):
        out = env.reset()
        if isinstance(out, tuple):
            obs, _ = out
        else:
            obs = out

        done = False
        truncated = False

        ep_total = 0.0
        ep_stage = np.zeros(2, dtype=np.float64)

        t = 0

        while not (done or truncated):
            action = echelon_base_stock_action(
                obs, env, S1, S2, max_action=max_action
            )

            step_out = env.step(action)

            if len(step_out) == 5:
                obs, reward, done, truncated, _ = step_out
            else:
                obs, reward, done, _ = step_out
                truncated = False

            r = np.asarray(reward, dtype=np.float64)

            # ✅ 关键：只在 warm-up 之后累计 reward
            if t >= warmup_periods:
                if r.ndim == 0:
                    ep_total += float(r)
                else:
                    ep_stage += r
                    ep_total += float(np.sum(r))

            t += 1

        total_returns.append(ep_total)
        stage_returns.append(ep_stage.copy())

    mean_total = float(np.mean(total_returns))
    mean_stage = np.mean(np.stack(stage_returns, axis=0), axis=0)
    return mean_total, mean_stage



# -----------------------------
# 网格搜索
# -----------------------------
@dataclass
class GridSpec:
    S1_values: np.ndarray
    S2_values: np.ndarray


def grid_search_S1S2(env_cfg: Dict[str, Any],
                     grid: GridSpec,
                     n_episodes: int = 50,
                     max_action: float = 40.0,
                     verbose: bool = True) -> Dict[str, Any]:
    best = {
        "S1": None,
        "S2": None,
        "mean_total_return": -np.inf,  # reward 越大越好（若是 -cost）
        "mean_stage_return": None,
    }

    results = []  # (S1, S2, mean_total, stage0, stage1)

    total_iters = len(grid.S1_values) * len(grid.S2_values)
    it = 0

    for S1 in grid.S1_values:
        for S2 in grid.S2_values:
            it += 1
            mean_total, mean_stage = evaluate_S1S2(
                env_cfg=env_cfg,
                S1=float(S1),
                S2=float(S2),
                n_episodes=n_episodes,
                max_action=max_action,
                warmup_periods=0,
            )
            results.append((float(S1), float(S2), mean_total, float(mean_stage[0]), float(mean_stage[1])))

            if mean_total > best["mean_total_return"]:
                best.update({
                    "S1": float(S1),
                    "S2": float(S2),
                    "mean_total_return": mean_total,
                    "mean_stage_return": mean_stage,
                })

            if verbose and (it % 10 == 0 or it == 1 or it == total_iters):
                print(f"[{it:4d}/{total_iters}] S1={S1:.2f}, S2={S2:.2f} | "
                      f"mean_total={mean_total:.4f} | per-stage={mean_stage} | "
                      f"best=({best['S1']},{best['S2']}) {best['mean_total_return']:.4f}")

    return {"best": best, "results": results}


# -----------------------------
# Example main
# -----------------------------
if __name__ == "__main__":
    # 环境配置：按你现在的 env 构造方式填
    env_cfg = dict(
        n_stages=2,
        lead_times=[2, 2],
        episode_length=100,
        render_mode=None,
    )

    # 网格范围（你可按经验缩小/放大）
    # 注意：S2 通常应 >= S1（不强制，但常见），你也可以用这个约束减少搜索量
    S1_values = np.arange(71, 75, 0.1)     # 0,2,4,...,80
    S2_values = np.arange(116, 120, 0.1)    # 0,2,4,...,160

    grid = GridSpec(S1_values=S1_values, S2_values=S2_values)

    out = grid_search_S1S2(
        env_cfg=env_cfg,
        grid=grid,
        n_episodes=100,     # 越大越稳但越慢
        max_action=40.0,
        verbose=True,
    )

    best = out["best"]
    print("\n===== BEST (S1,S2) FOUND =====")
    print(f"S1 = {best['S1']}, S2 = {best['S2']}")
    print(f"mean_total_return = {best['mean_total_return']:.6f}")
    print(f"mean_stage_return = {best['mean_stage_return']}")

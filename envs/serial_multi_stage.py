import numpy as np
import gymnasium as gym
from gymnasium import spaces


class DemandGenerator:
    """
    Simple stochastic demand generator for the downstream stage (stage 0).

    Default:
        Truncated Normal distribution N(mu, sigma^2) with lower bound at 0.
    This class can later be extended to AR(1), ARMA, seasonal models, etc.
    """

    def __init__(self, mu: float = 20.0, sigma: float = 5.0, seed: int | None = None):
        self.mu = mu
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)

    def reset(self):
        """Reset internal state if using AR(1)/ARMA demand models."""
        pass

    def sample(self) -> float:
        d = self.rng.normal(self.mu, self.sigma)
        return float(max(d, 0.0))


class SerialMultiStageEnv(gym.Env):
    """
    Serial multi-stage inventory system (Beer Game style), periodic review model.

    Stages are indexed as:
        0, 1, ..., n-1

    Interpretation:
        - Stage 0 = most downstream (retailer), faces external customer demand.
        - Stage n-1 = most upstream (manufacturer), assumed to have unlimited supply.

    -------------------------------------------------------------------------
    Core Periodic Flow (simplified Beer Game dynamics):
    -------------------------------------------------------------------------

    1. **Incoming shipments**
       Each stage receives its incoming shipment from the pipeline,
       delayed according to its lead time.

    2. **External customer demand (stage 0 only)**
       - Total demand = backlog[0] + external_demand_t
       - The stage ships as much as possible from inventory
       - Any unmet demand becomes backlog[0]

    3. **Internal serial demand propagation (stage 1..n-1)**
       For i >= 1:
           downstream_demand_i = backlog[i] + orders[i-1]
       - Stage i tries to satisfy downstream demand with its inventory
       - Unfilled demand becomes backlog[i]
       - Thus, upstream shortage directly restricts downstream service level

    4. **Order placement**
       Each stage places its replenishment order (action[i])
       to its upstream source. The order enters the stage's pipeline and
       will arrive after lead_time[i] periods.

    5. **Cost and reward**
       Total cost = Σ_i (holding_cost[i] * inventory[i]
                         + backlog_cost[i] * backlog[i])
       reward = -total_cost
       (This supports either centralized RL or multi-agent RL with a global reward.)

    -------------------------------------------------------------------------
    Simplifications in this version:
    -------------------------------------------------------------------------
    - Upstream supply (stage n-1) is assumed unlimited; only lead time matters.
    - Information has no delay (order information arrives instantaneously).
    - Backlog is explicitly modeled; negative inventory is not allowed.
    - This environment supports:
        * Single-agent centralized control
        * Multi-agent RL (each stage as an agent)
        * CTDE (centralized critic, decentralized actors)
        * Federated RL (local actors + federated critic updates)
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        n_stages: int = 2,
        lead_times: list[int] | None = None,
        holding_costs: list[float] | None = None,
        backlog_costs: list[float] | None = None,
        max_order: float = 40.0,
        episode_length: int = 100,
        demand_params: dict | None = None,
        render_mode: str | None = None,
        seed: int | None = None,
    ):
        super().__init__()

        assert n_stages >= 1, "n_stages must be >= 1"
        self.n_stages = n_stages
        self.render_mode = render_mode
        self.episode_length = episode_length
        self.time = 0

        # ---------------- Initialization of parameters ----------------
        if lead_times is None:
            self.lead_times = [1] * n_stages
        else:
            assert len(lead_times) == n_stages
            self.lead_times = list(lead_times)

        if holding_costs is None:
            self.holding_costs = [1.0] * n_stages
        else:
            assert len(holding_costs) == n_stages
            self.holding_costs = list(holding_costs)

        if backlog_costs is None:
            self.backlog_costs = [9.0] * n_stages
        else:
            assert len(backlog_costs) == n_stages
            self.backlog_costs = list(backlog_costs)

        self.max_order = float(max_order)

        # Demand generator for stage 0
        if demand_params is None:
            demand_params = {"mu": 20.0, "sigma": 5.0}
        self.demand_gen = DemandGenerator(
            mu=demand_params.get("mu", 20.0),
            sigma=demand_params.get("sigma", 5.0),
            seed=seed,
        )

        # ---------------- Gym Spaces ----------------
        self.max_lead_time = max(self.lead_times)

        # Observation structure:
        #   For each stage i:
        #     [inventory_i,
        #      backlog_i,
        #      pipeline vector (padded to max_lead_time),
        #      faced_demand_i]
        self.per_stage_obs_dim = 2 + self.max_lead_time + 1
        obs_dim = self.n_stages * self.per_stage_obs_dim

        high_obs = np.full(obs_dim, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-high_obs,
            high=high_obs,
            dtype=np.float32,
        )

        # Action: order quantity for each stage, continuous
        self.action_space = spaces.Box(
            low=0.0,
            high=self.max_order,
            shape=(self.n_stages,),
            dtype=np.float32,
        )

        # ---------------- Internal State ----------------
        self.inventory = None          # shape (n_stages,)
        self.backlog = None            # shape (n_stages,)
        self.pipeline = None           # list of arrays, each with length = lead_time[i]
        self.last_orders = None        # shape (n_stages,)
        self.rewards = None            # per-stage reward (global reward allocated)
        self.ship = None               # realized shipment <= actual order

        # RNG
        self.np_random, _ = gym.utils.seeding.np_random(seed)

    # ==================================================================
    # Gym API
    # ==================================================================
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.time = 0
        self.demand_gen.reset()
        customer_demand = 20  # initial placeholder

        # Initialize states
        self.inventory = np.ones(self.n_stages, dtype=np.float32) * 0
        self.backlog = np.zeros(self.n_stages, dtype=np.float32)
        self.last_orders = np.ones(self.n_stages, dtype=np.float32) * 20
        self.rewards = np.zeros(self.n_stages, dtype=np.float32)
        self.ship = np.zeros(self.n_stages, dtype=np.float32)

        # Initialize pipelines (if L=0, still allocate one slot)
        self.pipeline = []
        for L in self.lead_times:
            if L <= 0:
                self.pipeline.append(np.ones(1, dtype=np.float32) * 20)
            else:
                self.pipeline.append(np.ones(L, dtype=np.float32) * 20)

        obs = self._get_obs(customer_demand)
        info = {}
        return obs, info

    def step(self, action):
        """
        action: shape (n_stages,), order quantity for each stage.
        """
        action = np.asarray(action, dtype=np.float32)
        assert action.shape == (self.n_stages,)

        self.time += 1

        action_used = np.clip(action, 0.0, self.max_order)  # ✅ 关键：统一使用

        # 1) Receive shipments from pipeline
        self._receive_incoming_shipments()

        # 2) Customer demand at the downstream stage (stage 0)
        customer_demand = self.demand_gen.sample()

        # 3) Apply internal and external demands
        self._apply_all_demands(customer_demand, action_used)

        # 4) Place upstream orders (entering pipeline)
        self.last_orders = np.clip(action, 0.0, self.max_order)
        self._place_orders_to_upstream(self.last_orders)

        # 5) Compute cost and reward
        global_cost = self._compute_total_cost()
        global_reward = -global_cost

        obs = self._get_obs(customer_demand)
        terminated = False
        truncated = self.time >= self.episode_length

        info = {
            "time": self.time,
            "customer_demand": customer_demand,
            "cost": global_cost,
            "global_reward": global_reward,  # ✅ add
            "global_cost": global_cost,  # ✅ optional but helpful
        }

        if self.render_mode == "human":
            self.render()

        return obs, self.rewards, terminated, truncated, info

    def render(self):
        print(
            f"t={self.time}, inv={self.inventory}, "
            f"backlog={self.backlog}, last_orders={self.last_orders}"
        )

    def close(self):
        pass

    # ==================================================================
    # Internal Helper Functions
    # ==================================================================
    def _receive_incoming_shipments(self):
        """
        At the beginning of each period, each stage receives the leftmost
        pipeline element, then pipeline shifts left.
        """
        for i in range(self.n_stages):
            pipe = self.pipeline[i]
            arrival = pipe[0]
            pipe[:-1] = pipe[1:]
            pipe[-1] = 0.0
            self.pipeline[i] = pipe
            self.inventory[i] += arrival

    def _apply_all_demands(self, customer_demand: float, orders: np.ndarray):
        """
        Apply customer demand at stage 0, then propagate internal
        downstream orders upstream.

        For each stage i:

            effective_demand_i = backlog[i] + external_or_downstream_demand

            ship_i = min(inventory[i], effective_demand_i)
            inventory[i] -= ship_i
            backlog[i] = effective_demand_i - ship_i
        """

        # Stage 0: external customers
        total_demand_0 = self.backlog[0] + customer_demand
        ship_0 = min(self.inventory[0], total_demand_0)
        self.ship[0] = ship_0
        self.inventory[0] -= ship_0
        self.backlog[0] = total_demand_0 - ship_0

        # Stages 1..n-1: downstream demand = backlog[i] + orders[i-1]
        for i in range(1, self.n_stages):
            downstream_order = float(orders[i - 1])
            total_demand_i = self.backlog[i] + downstream_order

            ship_i = min(self.inventory[i], total_demand_i)
            self.ship[i] = ship_i
            self.inventory[i] -= ship_i
            self.backlog[i] = total_demand_i - ship_i

    def _place_orders_to_upstream(self, orders: np.ndarray):
        """
        Insert each stage's replenishment order into its pipeline tail.
        Shipment will arrive after lead_time periods.
        """
        for i in range(self.n_stages):
            pipe = self.pipeline[i]
            # pipe[-1] += float(orders[i])
            if i != self.n_stages - 1:
                pipe[-1] += self.ship[i + 1]
            else:
                pipe[-1] += float(orders[i])
            self.pipeline[i] = pipe

    # def _compute_total_cost(self) -> float:
    #     """
    #     Total cost = Σ_i (holding_cost_i * inventory_i + backlog_cost_i * backlog_i).
    #
    #     Backlog is explicitly stored; inventory is always non-negative.
    #     """
    #     global_cost = 0.0
    #     for i in range(self.n_stages):
    #         inv_pos = max(self.inventory[i], 0.0)
    #         back = max(self.backlog[i], 0.0)
    #
    #         # Per-stage reward (negative of per-stage cost)
    #         self.rewards[i] = -(self.holding_costs[i] * inv_pos +
    #                             self.backlog_costs[i] * back)
    #
    #         global_cost += self.holding_costs[i] * inv_pos + \
    #                        self.backlog_costs[i] * back
    #
    #     return float(global_cost)

    def _compute_total_cost(self) -> float:
        """
        Total cost (Clark–Scarf / Cachon–Zipkin style):

            total_cost
            = sum_i holding_cost_i * (on_hand_i + pipeline_proxy_i)
            + backlog_cost * customer_backlog   (ONLY ONCE)

        Notes:
        - customer_backlog = backlog[0]
        - internal backlogs (i>0) do NOT incur backlog cost
        - pipeline holding:
            * stage 0: no pipeline holding added (as requested)
            * stage i>0: add pipeline holding proxy using self.ship[i]
              (interpreted as pipeline/flow-related inventory proxy for upstream stages)
        - self.rewards is only for logging / per-stage decomposition
        """

        # ---------- 1) holding cost ----------
        holding_cost_total = 0.0

        for i in range(self.n_stages):
            on_hand = max(float(self.inventory[i]), 0.0)

            # pipeline holding proxy: only for upstream stages (i>0)
            if i > 0:
                # pipe_proxy = max(float(self.ship[i]), 0.0)
                pipe_proxy = max(np.sum(self.pipeline[i - 1]), 0.0)
            else:
                pipe_proxy = 0.0

            holding_cost_total += self.holding_costs[i] * (on_hand + pipe_proxy)

        # ---------- 2) customer backlog cost (ONLY ONCE) ----------
        customer_backlog = max(float(self.backlog[0]), 0.0)
        backlog_cost_total = self.backlog_costs[0] * customer_backlog

        # ---------- 3) global cost ----------
        global_cost = holding_cost_total + backlog_cost_total

        # ---------- 4) per-stage rewards (FOR LOGGING ONLY) ----------
        # stage i: negative holding cost component (including pipeline proxy for i>0)
        for i in range(self.n_stages):
            on_hand = max(float(self.inventory[i]), 0.0)
            if i > 0:
                # pipe_proxy = max(float(self.ship[i]), 0.0)
                pipe_proxy = max(np.sum(self.pipeline[i - 1]), 0.0)
            else:
                pipe_proxy = 0.0

            self.rewards[i] = -(self.holding_costs[i] * (on_hand + pipe_proxy))

        # assign ALL backlog cost to stage 0 (for interpretability)
        self.rewards[0] -= backlog_cost_total

        return float(global_cost)

    def _get_obs(self, customer_demand) -> np.ndarray:
        """
        Construct global observation vector by concatenating per-stage states.

        For each stage i:
            [inventory_i,
             backlog_i,
             pipeline vector (padded to max_lead_time),
             faced_demand_i]

        faced_demand_i =
            - external demand for stage 0
            - last_orders[i-1] for stage i>=1
        """
        obs_per_stage = []
        for i in range(self.n_stages):
            inv = self.inventory[i]
            back = self.backlog[i]
            pipe = self.pipeline[i]

            # Pad pipeline sequence to uniform length
            if len(pipe) < self.max_lead_time:
                pad_width = self.max_lead_time - len(pipe)
                pipe_vec = np.concatenate(
                    [pipe, np.zeros(pad_width, dtype=np.float32)]
                )
            else:
                pipe_vec = pipe.copy()

            if i == 0:
                faced_demand = customer_demand
            else:
                faced_demand = self.last_orders[i - 1]

            vec = np.concatenate([[inv], [back], pipe_vec, [faced_demand]])
            obs_per_stage.append(vec)

        obs = np.concatenate(obs_per_stage).astype(np.float32)
        return obs

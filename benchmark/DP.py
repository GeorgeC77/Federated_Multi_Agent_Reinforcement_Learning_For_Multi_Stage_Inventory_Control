import numpy as np

# ============================================
# Parameters (modifiable)
# ============================================

holding_cost = 1.0
backlog_cost = 9.0
discount_factor = 0.99     # γ < 1 ensures convergence
max_S = 60                 # search range of candidate base-stock levels
inventory_min = -30        # lower bound of state space (backlog)
inventory_max = 30        # upper bound of state space
state_space = range(inventory_min, inventory_max + 1)

# Demand distribution (example: discrete normal-like distribution)
# You can replace this with empirical distribution or Poisson, etc.
demand_prob = {
    0: 0.05,
    1: 0.10,
    2: 0.15,
    3: 0.25,
    4: 0.25,
    5: 0.15,
    6: 0.05
}

# ============================================
# Helper Functions
# ============================================

def next_state(x, S, demand):
    """ Transition: x' = x + order - demand, where order = max(0, S - x). """
    order = max(0, S - x)
    x_next = x + order - demand
    return min(max(x_next, inventory_min), inventory_max)

def stage_cost(x):
    """ Holding + backlog cost at state x. """
    return holding_cost * max(x, 0) + backlog_cost * max(-x, 0)

def expected_cost(x, S, V):
    """ Expected cost-to-go for state x under candidate target S. """
    total = 0
    for d, p in demand_prob.items():
        x_next = next_state(x, S, d)
        total += p * (stage_cost(x_next) + discount_factor * V[x_next])
    return total


# ============================================
# Value Iteration
# ============================================

def solve_dp(tolerance=1e-4, max_iter=500):
    V = {x: 0 for x in state_space}  # initialize value function
    policy = {x: 0 for x in state_space}

    for iteration in range(max_iter):
        V_new = {}
        delta = 0

        for x in state_space:
            # Test all possible base-stock levels S and pick the best
            cost_candidates = [expected_cost(x, S, V) for S in range(max_S + 1)]
            best_cost = min(cost_candidates)
            best_S = np.argmin(cost_candidates)

            V_new[x] = best_cost
            policy[x] = best_S

            delta = max(delta, abs(V_new[x] - V[x]))

        V = V_new

        # Convergence check
        if delta < tolerance:
            print(f"DP converged after {iteration+1} iterations, Δ={delta:.6f}")
            break

    return V, policy


# ============================================
# Run DP Solver
# ============================================

V, policy = solve_dp()

# ============================================
# Extract global optimal base-stock level
# ============================================

# Since base-stock policy is threshold-based, the optimal S* is the most common or stable value in policy
values, counts = np.unique(list(policy.values()), return_counts=True)
S_opt = values[np.argmax(counts)]

print("\nOptimal base-stock level (DP result):", S_opt)
print("\nExample of policy mapping:")
for x in range(-10, 21):
    print(f"x={x:3d} → S={policy[x]}")

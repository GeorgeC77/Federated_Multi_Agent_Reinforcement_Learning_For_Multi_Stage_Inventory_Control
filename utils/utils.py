import numpy as np


class RunningNormalizer:
    """
    Maintains running mean and variance for normalization.
    Supports:
        update(x): update running statistics with observation x
        normalize(x): return normalized version
        get_stats(): return mean/var (for federated averaging)
        set_stats(mean, var): load aggregated normalization parameters
    """

    def __init__(self, size, epsilon=1e-6):
        self.size = size
        self.epsilon = epsilon

        # Running statistics
        self.mean = np.zeros(size, dtype=np.float32)
        self.var = np.ones(size, dtype=np.float32)
        self.count = epsilon  # avoid division by zero

    def update(self, x):
        """Update statistics with a new observation."""
        x = np.array(x, dtype=np.float32)

        # Welford update for numerically stable mean/variance
        batch_mean = x
        batch_var = np.zeros_like(x)
        batch_count = 1

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        # Update mean
        new_mean = self.mean + delta * batch_count / total_count

        # Update variance
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        updated_var = (m_a + m_b + delta * delta * self.count * batch_count / total_count) / total_count

        self.mean = new_mean
        self.var = updated_var
        self.count = total_count

    def normalize(self, x):
        """Return normalized observation."""
        x = np.array(x, dtype=np.float32)
        return (x - self.mean) / (np.sqrt(self.var) + self.epsilon)

    def get_stats(self):
        """Export stats (for federated averaging)."""
        return {
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": self.count,
        }

    def set_stats(self, mean, var, count=None):
        """Import federated aggregated stats."""
        self.mean = np.array(mean, dtype=np.float32)
        self.var = np.array(var, dtype=np.float32)
        if count is not None:
            self.count = count  # optional

"""Shared pieces: feature list, traffic simulator, and the Detector wrapper."""
import numpy as np

FEATURES = ["duration", "pkt_count", "byte_count", "mean_pkt_size",
            "std_pkt_size", "mean_iat", "syn_ratio", "proto"]

# Synthetic traffic profiles (used for benchmarking when no real data is available)
PROFILES = {
    "normal":  dict(dur=2.0,  pkts=20,  size=120,  sd=25,  syn=0.03, proto=[0.8, 0.2]),
    "dos":     dict(dur=1.0,  pkts=500, size=60,   sd=3,   syn=0.90, proto=[1.0, 0.0]),
    "scan":    dict(dur=0.05, pkts=2,   size=44,   sd=1,   syn=0.98, proto=[1.0, 0.0]),
    "exfil":   dict(dur=30.0, pkts=800, size=1300, sd=100, syn=0.01, proto=[1.0, 0.0]),
    # never used for training: acts as the "unknown attack" family
    "unknown": dict(dur=10.0, pkts=60,  size=600,  sd=300, syn=0.20, proto=[0.3, 0.7]),
}


def simulate(kind, n, seed=None):
    rng = np.random.default_rng(seed)
    p = PROFILES[kind]
    duration = rng.lognormal(np.log(p["dur"]), 0.4, n)
    pkts = rng.poisson(p["pkts"], n) + 1
    mean_size = np.clip(rng.normal(p["size"], p["size"] * 0.15, n), 40, 1500)
    std_size = np.abs(rng.normal(p["sd"], p["sd"] * 0.2 + 0.5, n))
    syn = np.clip(rng.normal(p["syn"], 0.02, n), 0, 1)
    proto = rng.choice([1, 2], size=n, p=p["proto"])
    return np.column_stack([duration, pkts, pkts * mean_size, mean_size,
                            std_size, duration / pkts, syn, proto])


class Detector:
    """Scaler + model + threshold. score() is always 'higher = more anomalous'."""

    def __init__(self, name, model, scaler, threshold=0.0):
        self.name, self.model, self.scaler, self.threshold = name, model, scaler, threshold

    @staticmethod
    def prep(X):
        X = np.array(np.atleast_2d(X), dtype=float)
        X[:, :6] = np.log1p(np.clip(X[:, :6], 0, None))  # tame heavy-tailed features
        return X

    def score(self, X):
        Z = self.scaler.transform(self.prep(X))
        if self.name == "OCSVM":
            return -self.model.decision_function(Z)
        if self.name == "MCD":
            return self.model.mahalanobis(Z)
        return -self.model.score_samples(Z)  # LOF, KDE, GMM, IsolationForest

    def predict(self, X):
        return (self.score(X) > self.threshold).astype(int)

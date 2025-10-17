import numpy as np

def generate_knapsack_dataset(
    n_items=60,
    capacity_ratio=0.3,
    n_heavy=10,
    n_efficient=8,
    n_deceptive=5,
    value_range=(10, 400),
    weight_range=(1, 50),
    heavy_value_mult=2,
    heavy_weight_mult=3,
    efficient_value_mult=2,
    efficient_weight_div=2,
    deceptive_value_div=3,
    deceptive_weight_mult=2,
    seed=None
):
    if seed is not None:
        np.random.seed(seed)

    values = np.random.randint(*value_range, n_items)
    weights = np.random.randint(*weight_range, n_items)
    all_idx = np.arange(n_items)

    heavy_idx = np.random.choice(all_idx, min(n_heavy, n_items), replace=False)
    remaining = np.setdiff1d(all_idx, heavy_idx)
    efficient_idx = np.random.choice(remaining, min(n_efficient, len(remaining)), replace=False)
    remaining = np.setdiff1d(remaining, efficient_idx)
    deceptive_idx = np.random.choice(remaining, min(n_deceptive, len(remaining)), replace=False)

    values[heavy_idx] *= heavy_value_mult
    weights[heavy_idx] *= heavy_weight_mult
    values[efficient_idx] *= efficient_value_mult
    weights[efficient_idx] //= efficient_weight_div
    weights[weights == 0] = 1
    values[deceptive_idx] //= deceptive_value_div
    weights[deceptive_idx] *= deceptive_weight_mult

    capacity = int(np.sum(weights) * capacity_ratio)
    info = {"heavy": heavy_idx, "efficient": efficient_idx, "deceptive": deceptive_idx}
    return values, weights, capacity, info

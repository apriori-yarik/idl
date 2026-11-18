import numpy as np

def initialize_weights(low, high, size, seed=42):
    rand_gen = np.random.default_rng(seed)

    return rand_gen.uniform(low, high, size)
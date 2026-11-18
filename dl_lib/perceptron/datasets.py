import numpy as np

def create_dataset(n: int) -> list[tuple[float, float]]:
    x = np.arange(n, dtype=float)
    y = x * 2

    return list(zip(x.tolist(), y.tolist()))

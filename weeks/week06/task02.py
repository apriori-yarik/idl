from dl_lib.optim.optimizers import SGD
import torch

def main() -> int:
    p = torch.tensor([1., 2.])
    p.grad = torch.tensor([0.5, 0.5])

    sgd = SGD([p], 0.1)

    print(f"Before step: p = {p}")
    sgd.step()
    print(f"After step:  p = {p}")
    return 0


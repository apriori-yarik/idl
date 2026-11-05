from dl_lib.optim.optimizer import Optimizer
import torch

class SGD(Optimizer):
    def step(self):
        with torch.no_grad():
            for w in self.params:
                w -= self.lr * w.grad

        
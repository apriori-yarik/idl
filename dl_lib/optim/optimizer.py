from abc import ABC, abstractmethod

class Optimizer(ABC):
    def __init__(self, parameters, lr):
        self.params = parameters
        self.lr = lr

    @abstractmethod
    def step(self):
        pass

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()
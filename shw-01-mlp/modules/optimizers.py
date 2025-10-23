import numpy as np
from typing import Tuple
from .base import Module, Optimizer


class SGD(Optimizer):
    """
    Optimizer implementing stochastic gradient descent with momentum
    """
    def __init__(self, module: Module, lr: float = 1e-2, momentum: float = 0.0,
                 weight_decay: float = 0.0):
        """
        :param module: neural network containing parameters to optimize
        :param lr: learning rate
        :param momentum: momentum coefficient (alpha)
        :param weight_decay: weight decay (L2 penalty)
        """
        super().__init__(module)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

    def step(self):
        parameters = self.module.parameters()
        gradients = self.module.parameters_grad()
        if 'm' not in self.state:
            self.state['m'] = [np.zeros_like(param) for param in parameters]

        for i, (param, grad) in enumerate(zip(parameters, gradients)):
            if grad is None:
                continue

            # add L2 penalty
            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * param

            m = self.state['m'][i]

            # update momentum: m = momentum*m - lr*grad
            np.multiply(self.momentum, m, out=m)       # m *= momentum
            np.add(m, -self.lr * grad, out=m)          # m -= lr * grad

            # update parameters: param += m
            np.add(param, m, out=param)


class Adam(Optimizer):
    """
    Optimizer implementing Adam
    """
    def __init__(self, module: Module, lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0):
        """
        :param module: neural network containing parameters to optimize
        :param lr: learning rate
        :param betas: Adam beta1 and beta2
        :param eps: Adam eps
        :param weight_decay: weight decay (L2 penalty)
        """
        super().__init__(module)
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.weight_decay = weight_decay

    def step(self):
        parameters = self.module.parameters()
        gradients = self.module.parameters_grad()

        if 'm' not in self.state:
            self.state['m'] = [np.zeros_like(param) for param in parameters]
            self.state['v'] = [np.zeros_like(param) for param in parameters]
            self.state['t'] = 0

        self.state['t'] += 1
        t = self.state['t']

        for i, (param, grad) in enumerate(zip(parameters, gradients)):
            if grad is None:
                continue

            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * param

            m = self.state['m'][i]
            v = self.state['v'][i]

            # update biased first moment
            np.multiply(self.beta1, m, out=m)                 # m = beta1*m
            np.add(m, (1 - self.beta1) * grad, out=m)         # m += (1-beta1)*grad

            # update biased second moment
            np.multiply(self.beta2, v, out=v)                 # v = beta2*v
            np.add(v, (1 - self.beta2) * (grad * grad), out=v)# v += (1-beta2)*grad^2

            # bias correction
            m_hat = m / (1 - self.beta1 ** t)
            v_hat = v / (1 - self.beta2 ** t)

            # parameter update
            np.add(param, -self.lr * m_hat / (np.sqrt(v_hat) + self.eps), out=param)

import numpy as np
from typing import List
from .base import Module


class Linear(Module):
    """
    Applies linear (affine) transformation of data: y = x W^T + b
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        :param in_features: input vector features
        :param out_features: output vector features
        :param bias: whether to use additive bias
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.random.uniform(-1, 1, (out_features, in_features)) / np.sqrt(in_features)
        self.bias = np.random.uniform(-1, 1, out_features) / np.sqrt(in_features) if bias else None

        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias) if bias else None

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of shape (batch_size, in_features)
        :return: array of shape (batch_size, out_features)
        """
        output = input @ self.weight.T
        if self.bias is not None:
            output += self.bias
        return output

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of shape (batch_size, in_features)
        :param grad_output: array of shape (batch_size, out_features)
        :return: array of shape (batch_size, in_features)
        """
        return grad_output @ self.weight

    def update_grad_parameters(self, input: np.ndarray, grad_output: np.ndarray):
        """
        :param input: array of shape (batch_size, in_features)
        :param grad_output: array of shape (batch_size, out_features)
        """
        self.grad_weight += grad_output.T @ input
        if self.bias is not None:
            self.grad_bias += grad_output.sum(axis=0)

    def zero_grad(self):
        self.grad_weight.fill(0)
        if self.bias is not None:
            self.grad_bias.fill(0)

    def parameters(self) -> List[np.ndarray]:
        if self.bias is not None:
            return [self.weight, self.bias]

        return [self.weight]

    def parameters_grad(self) -> List[np.ndarray]:
        if self.bias is not None:
            return [self.grad_weight, self.grad_bias]

        return [self.grad_weight]

    def __repr__(self) -> str:
        out_features, in_features = self.weight.shape
        return f'Linear(in_features={in_features}, out_features={out_features}, ' \
            f'bias={not self.bias is None})'


class BatchNormalization(Module):
    """
    Applies batch normalization transformation
    """
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        # статистики
        self.running_mean = np.zeros(num_features, dtype=float)
        self.running_var = np.ones(num_features, dtype=float)

        # параметры (γ и β)
        self.weight = np.ones(num_features, dtype=float) if affine else None
        self.bias = np.zeros(num_features, dtype=float) if affine else None

        # градиенты
        self.grad_weight = np.zeros_like(self.weight) if affine else None
        self.grad_bias = np.zeros_like(self.bias) if affine else None

        # промежуточные значения для backward
        self.mean = None
        self.var = None
        self.inv_sqrt_var = None
        self.norm_input = None
        self.input_centered = None

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        Forward pass
        :param input: shape (batch_size, num_features)
        :return: normalized output
        """
        if self.training:
            # batch statistics
            self.mean = input.mean(axis=0)
            self.var = input.var(axis=0)

            # update running stats
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * self.mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * self.var

            # normalize
            self.input_centered = input - self.mean
            self.inv_sqrt_var = 1.0 / np.sqrt(self.var + self.eps)
            self.norm_input = self.input_centered * self.inv_sqrt_var
        else:
            # use running statistics
            self.input_centered = input - self.running_mean
            self.inv_sqrt_var = 1.0 / np.sqrt(self.running_var + self.eps)
            self.norm_input = self.input_centered * self.inv_sqrt_var

        out = self.norm_input
        if self.affine:
            out = self.weight * out + self.bias
        return out

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass: gradient w.r.t. input
        """
        N, D = input.shape

        if not self.training:
            # в режиме eval градиенты просто проходят через scale+shift
            return grad_output * (self.weight if self.affine else 1.0) * self.inv_sqrt_var

        # grad w.r.t. normalized input
        grad_norm = grad_output * (self.weight if self.affine else 1.0)

        # формулы batchnorm backward
        dvar = np.sum(grad_norm * self.input_centered * -0.5 * self.inv_sqrt_var**3, axis=0)
        dmean = np.sum(grad_norm * -self.inv_sqrt_var, axis=0) + dvar * np.mean(-2.0 * self.input_centered, axis=0)

        grad_input = grad_norm * self.inv_sqrt_var + dvar * 2.0 * self.input_centered / N + dmean / N
        return grad_input

    def update_grad_parameters(self, input: np.ndarray, grad_output: np.ndarray):
        """
        Gradient w.r.t. weight and bias
        """
        if not self.affine:
            return

        if self.training:
            grad_norm = grad_output * self.norm_input
            self.grad_weight += grad_norm.sum(axis=0)
            self.grad_bias += grad_output.sum(axis=0)
        else:
            # при eval статистика фиксирована, но всё равно считаем градиенты
            grad_norm = grad_output * self.norm_input
            self.grad_weight += grad_norm.sum(axis=0)
            self.grad_bias += grad_output.sum(axis=0)

    def zero_grad(self):
        if self.affine:
            self.grad_weight.fill(0)
            self.grad_bias.fill(0)

    def parameters(self) -> List[np.ndarray]:
        return [self.weight, self.bias] if self.affine else []

    def parameters_grad(self) -> List[np.ndarray]:
        return [self.grad_weight, self.grad_bias] if self.affine else []

    def __repr__(self) -> str:
        return f'BatchNormalization(num_features={len(self.running_mean)}, eps={self.eps}, momentum={self.momentum}, affine={self.affine})'




class Dropout(Module):
    """
    Applies dropout transformation
    """
    def __init__(self, p=0.5):
        super().__init__()
        assert 0 <= p < 1
        self.p = p
        self.mask = None

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        if self.training:
            self.mask = (np.random.rand(*input.shape) >= self.p).astype(float) / (1.0 - self.p)
            output = self.mask * input
        else:
            output = input
        return output

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        if self.training:
            output = self.mask * grad_output
            return output
        else:
            return grad_output

    def __repr__(self) -> str:
        return f'Dropout(p={self.p})'


class Sequential(Module):
    """
    Container for consecutive application of modules
    """
    def __init__(self, *args):
        super().__init__()
        self.modules = list(args)

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        output = input
        for layer in self.modules:
            output = layer.forward(output)   # важно forward, а не compute_output
        return output

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        # сохранить все промежуточные выходы
        out = input
        outputs = [input]
        for layer in self.modules:
            out = layer.forward(out)
            outputs.append(out)

        # обратный проход
        grad_in = grad_output
        for i in reversed(range(len(self.modules))):
            grad_in = self.modules[i].backward(outputs[i], grad_in)

        return grad_in

    def __getitem__(self, item):
        return self.modules[item]

    def train(self):
        for module in self.modules:
            module.train()

    def eval(self):
        for module in self.modules:
            module.eval()

    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()

    def parameters(self) -> List[np.ndarray]:
        return [parameter for module in self.modules for parameter in module.parameters()]

    def parameters_grad(self) -> List[np.ndarray]:
        return [grad for module in self.modules for grad in module.parameters_grad()]

    def __repr__(self) -> str:
        repr_str = 'Sequential(\n'
        for module in self.modules:
            repr_str += ' ' * 4 + repr(module) + '\n'
        repr_str += ')'
        return repr_str

import numpy as np
from .base import Criterion
from .activations import LogSoftmax

class MSELoss(Criterion):
    """
    Mean squared error criterion
    """
    def compute_output(self, input: np.ndarray, target: np.ndarray) -> float:
        assert input.shape == target.shape, 'input and target shapes not matching'
        loss = np.mean((input - target) ** 2)
        return np.float64(loss)

    def compute_grad_input(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        assert input.shape == target.shape, 'input and target shapes not matching'
        grad = 2 * (input - target) / input.size
        return grad


class CrossEntropyLoss(Criterion):
    """
    Cross-entropy criterion over distribution logits
    """
    def __init__(self):
        super().__init__()
        self.log_softmax = LogSoftmax()

    def compute_output(self, input: np.ndarray, target: np.ndarray) -> float:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: loss value
        """
        batch_size = input.shape[0]
        log_probs = self.log_softmax.forward(input)  # shape: (batch_size, num_classes)
        loss = -np.sum(log_probs[np.arange(batch_size), target]) / batch_size
        return np.float64(loss)

    def compute_grad_input(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        batch_size = input.shape[0]
        probs = np.exp(self.log_softmax.forward(input))  # softmax probabilities
        probs[np.arange(batch_size), target] -= 1  # dL/dz = softmax - 1 for correct class
        grad = probs / batch_size
        return grad




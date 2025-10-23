import numpy as np
from .base import Module
from scipy.special import expit, softmax, log_softmax


class ReLU(Module):
    """
    Applies element-wise ReLU function
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        return np.maximum(0, input)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        def relu_derivative(x):
            return (x > 0).astype(float)
        return grad_output * relu_derivative(input)


class Sigmoid(Module):
    """
    Applies element-wise sigmoid function
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        return expit(input)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        return grad_output * expit(input)*(1 - expit(input))


class Softmax(Module):
    """
    Applies Softmax operator over the last dimension
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        return softmax(input, axis=1)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        s = softmax(input, axis=1)
        
        # Векторизованный расчёт через einsum
        # s[:,:,None] создаёт 3D тензор (batch, classes, 1)
        # s[:,None,:] создаёт 3D тензор (batch, 1, classes)
        jacobian = np.einsum('ij,ik->ijk', s, s)  # внешнее произведение
        diag = np.einsum('ij,jk->ijk', s, np.eye(s.shape[1]))  # диагональные элементы
        jacobian = diag - jacobian
        
        # Умножение Якоби на градиент для каждого примера в батче
        return np.einsum('ijk,ik->ij', jacobian, grad_output)


class LogSoftmax(Module):
    """
    Applies LogSoftmax operator over the last dimension
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        return log_softmax(input, axis=1)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        Вычисляет градиент по входу для LogSoftmax
        
        :param input: array размера (batch_size, num_classes)
        :param grad_output: array того же размера (градиенты потерь по выходам log_softmax)
        :return: градиент по входу, array размера (batch_size, num_classes)
        """
        s = softmax(input, axis=1)  # (batch_size, num_classes)
        
        # Градиент для log(softmax(x))_i по x_j:
        # ∂L/∂x_j = ∂L/∂log_s_i * (δ_ij - s_j)
        # где δ_ij - символ Кронекера (1 если i=j, иначе 0)
        
        # Векторизованная реализация:
        # sum_grad = sum(grad_output, axis=1, keepdims=True)
        # grad_input = grad_output - s * sum_grad
        
        # Более стабильная реализация:
        sum_grad = np.sum(grad_output, axis=1, keepdims=True)
        return grad_output - s * sum_grad




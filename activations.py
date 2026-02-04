from tensor import Tensor
import numpy as np

class ReLU:
    def forward(self, x: Tensor) -> Tensor:
        """apply ReLU activation element-wise.
           ReLU (Rectified Linear Unit) is deceptively simple: 
           it zeros out negative values and leaves positive values unchanged. """
        result = np.maximum(0, x.data)
        return Tensor(result)

class Sigmoid:
    def forward(self, x: Tensor) -> Tensor:
        """apply sigmoid activation element-wise.
           sigmoid maps any real number to the range (0, 1), 
           making it perfect for representing probabilities"""
        z = np.clip(x.data, -500, 500)  # prevent overflow
        result_data = np.zeros_like(z)

        # positive values: 1 / (1 + exp(-x))
        pos_mask = z >= 0
        result_data[pos_mask] = 1.0 / (1.0 + np.exp(-z[pos_mask]))

        # negative values: exp(x) / (1 + exp(x))
        neg_mask = z < 0
        exp_z = np.exp(z[neg_mask])
        result_data[neg_mask] = exp_z / (1.0 + exp_z)

        return Tensor(result_data)

class Tanh:
    def forward(self, x: Tensor) -> Tensor:
        """apply tanh activation element-wise."""
        result = np.tanh(x.data)
        return Tensor(result)

class Softmax:
    def forward(self, x: Tensor, dim: int = -1) -> Tensor:
        """apply softmax activation along specified dimension."""
        # numerical stability: subtract max to prevent overflow
        x_max_data = np.max(x.data, axis=dim, keepdims=True)
        x_max = Tensor(x_max_data, requires_grad=False)
        x_shifted = x - x_max

        # compute exponentials
        exp_values = Tensor(np.exp(x_shifted.data), requires_grad=x_shifted.requires_grad)

        # sum along dimension
        exp_sum_data = np.sum(exp_values.data, axis=dim, keepdims=True)
        exp_sum = Tensor(exp_sum_data, requires_grad=exp_values.requires_grad)

        # normalize to get probabilities
        result = exp_values / exp_sum
        return result

import numpy as np
from constants import *
from tensor import Tensor
from activations import *

class Layer:
    """
    Base interface for all neural network layers.

    All layers should inherit from this class and implement:
    - forward(x): compute layer output
    - parameters(): return list of trainable parameters

    The __call__ method is provided to make layers callable.
    """ 

    def forward(self, x):
        """
        forward pass through the layer.

        args:
            x: input tensor

        returns:
            output tensor after transformation
        """
        raise NotImplementedError("Subclasses must implement forward()")

    def __call__(self, x, *args, **kwargs):
        """allow layer to be called like a function."""
        return self.forward(x, *args, **kwargs)

    def parameters(self):
        """
        return list of trainable parameters.

        returns:
            list of tensor objects (weights and biases)
        """
        return []  # base class has no parameters

    def __repr__(self):
        """string representation of the layer."""
        return f"{self.__class__.__name__}()"
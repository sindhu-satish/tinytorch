"""
tensor class implementation - a numpy-based tensor library similar to pytorch.
provides a simple interface for multi-dimensional array operations.
"""
import numpy as np
from constants import *


class Tensor:
    """
    a multi-dimensional array tensor backed by numpy.
    provides basic arithmetic operations, matrix multiplication, and transformations.
    """
    
    def __init__(self, data):
        """
        initialize a tensor from array-like data.
        
        args:
            data: array-like data (list, tuple, numpy array, etc.) to convert to tensor.
                  automatically converted to float32 numpy array.
        """
        self.data = np.array(data, dtype=np.float32)
        self.shape = self.data.shape
        self.size = self.data.size
        self.dtype = self.data.dtype

    def __repr__(self):
        """string representation for developers (includes shape info)."""
        return f"Tensor(data={self.data}, shape={self.shape})"

    def __str__(self):
        """human-readable string representation."""
        return f"Tensor({self.data})"

    def numpy(self):
        """
        return the underlying numpy array.
        
        returns:
            numpy.ndarray: the numpy array backing this tensor.
        """
        return self.data

    def memory_footprint(self):
        """
        calculate the memory footprint of the tensor in bytes.
        
        returns:
            int: number of bytes occupied by the tensor data.
        """
        return self.data.nbytes

    def __add__(self, other):
        """element-wise addition. supports tensor or scalar addition."""
        if isinstance(other, Tensor):
            return Tensor(self.data + other.data)
        else:
            return Tensor(self.data + other)

    def __sub__(self, other):
        """element-wise subtraction. supports tensor or scalar subtraction."""
        if isinstance(other, Tensor):
            return Tensor(self.data - other.data)
        else:
            return Tensor(self.data - other)

    def __mul__(self, other):
        """element-wise multiplication. supports tensor or scalar multiplication."""
        if isinstance(other, Tensor):
            return Tensor(self.data * other.data)
        else:
            return Tensor(self.data * other)

    def __truediv__(self, other):
        """element-wise division. supports tensor or scalar division."""
        if isinstance(other, Tensor):
            return Tensor(self.data / other.data)
        else:
            return Tensor(self.data / other)

    def matmul(self, other):
        """
        matrix multiplication (matrix product) of two tensors.
        for 2d tensors, performs standard matrix multiplication.
        for higher dimensions, uses numpy's batched matrix multiplication.
        
        args:
            other: tensor to multiply with (must be a tensor instance).
            
        returns:
            tensor: result of matrix multiplication.
            
        raises:
            typeerror: if other is not a tensor.
            valueerror: if tensor shapes are incompatible for matrix multiplication.
        """
        if not isinstance(other, Tensor):
            raise TypeError(f"Expected Tensor for matrix multiplication, got {type(other)}")
        
        # handle scalar case (0-dimensional tensors)
        if self.shape == () or other.shape == ():
            return Tensor(self.data * other.data)
        if len(self.shape) == 0 or len(other.shape) == 0:
            return Tensor(self.data * other.data)
        
        # validate matrix multiplication dimensions for 2d+ tensors
        if len(self.shape) >= 2 and len(other.shape) >= 2:
            if self.shape[-1] != other.shape[-2]:
                raise ValueError(
                    f"Cannot perform matrix multiplication: {self.shape} @ {other.shape}. "
                    f"Inner dimensions must match: {self.shape[-1]} ≠ {other.shape[-2]}"
                )

        a = self.data
        b = other.data
        
            # explicit loop-based implementation for 2d case (educational clarity)
        if len(a.shape) == 2 and len(b.shape) == 2:
            M, K = a.shape
            K2, N = b.shape
            result_data = np.zeros((M, N), dtype=a.dtype)

            # compute matrix product element by element
            for i in range(M):
                for j in range(N):
                    result_data[i, j] = np.dot(a[i, :], b[:, j])
        else:
            # use numpy's optimized matmul for higher dimensions or broadcasting
            result_data = np.matmul(a, b)

        return Tensor(result_data)

    def __matmul__(self, other):
        """enable @ operator for matrix multiplication (e.g., a @ b)."""
        return self.matmul(other)

    def __getitem__(self, key):
        """
        indexing and slicing support.
        
        args:
            key: index, slice, or tuple of indices/slices.
            
        returns:
            tensor: new tensor with indexed data.
        """
        result_data = self.data[key]
        # ensure result is always a numpy array (handles scalar indexing)
        if not isinstance(result_data, np.ndarray):
            result_data = np.array(result_data)
        return Tensor(result_data)

    def reshape(self, *shape):
        """
        reshape the tensor to a new shape.
        total number of elements must remain the same.
        supports -1 for automatic dimension inference.
        
        args:
            *shape: new shape dimensions. can be passed as separate arguments
                   or as a single tuple/list. use -1 for one unknown dimension.
                   
        returns:
            tensor: reshaped tensor.
            
        raises:
            valueerror: if total elements don't match or multiple -1 values are provided.
        """
        # handle both reshape(2, 3) and reshape((2, 3)) syntax
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            new_shape = tuple(shape[0])
        else:
            new_shape = shape
        
        # handle -1 placeholder for automatic dimension inference
        if -1 in new_shape:
            if new_shape.count(-1) > 1:
                raise ValueError("Can only specify one unknown dimension with -1")
            
            # calculate the unknown dimension size
            known_size = 1
            unknown_idx = new_shape.index(-1)
            for i, dim in enumerate(new_shape):
                if i != unknown_idx:
                    known_size *= dim
            unknown_dim = self.size // known_size
            
            # replace -1 with calculated dimension
            new_shape = list(new_shape)
            new_shape[unknown_idx] = unknown_dim
            new_shape = tuple(new_shape)
        
        # validate that total elements match
        if np.prod(new_shape) != self.size:
            target_size = int(np.prod(new_shape))
            raise ValueError(
                f"Total elements must match: {self.size} ≠ {target_size}"
            )
        
        reshaped_data = np.reshape(self.data, new_shape)
        return Tensor(reshaped_data)

    def transpose(self, dim0=None, dim1=None):
        """
        transpose the tensor.
        if no arguments provided, swaps the last two dimensions (standard transpose).
        if dim0 and dim1 provided, swaps those specific dimensions.
        
        args:
            dim0: first dimension to swap (optional).
            dim1: second dimension to swap (optional).
            
        returns:
            tensor: transposed tensor.
            
        raises:
            valueerror: if only one of dim0/dim1 is specified.
        """
        if dim0 is None and dim1 is None:
            # standard transpose: swap last two dimensions
            if len(self.shape) < 2:
                # no-op for 1d or 0d tensors
                return Tensor(self.data.copy())
            else:
                axes = list(range(len(self.shape)))
                axes[-2], axes[-1] = axes[-1], axes[-2]
                transposed_data = np.transpose(self.data, axes)
        else:
            # swap specific dimensions
            if dim0 is None or dim1 is None:
                raise ValueError("Both dim0 and dim1 must be specified")
            axes = list(range(len(self.shape)))
            axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
            transposed_data = np.transpose(self.data, axes)
        return Tensor(transposed_data)

    def sum(self, axis=None, keepdims=False):
        """
        sum elements along specified axis(es).
        
        args:
            axis: axis or axes along which to sum. none sums all elements.
            keepdims: if true, keeps reduced dimensions with size 1.
            
        returns:
            tensor: sum result.
        """
        result = np.sum(self.data, axis=axis, keepdims=keepdims)
        return Tensor(result)

    def mean(self, axis=None, keepdims=False):
        """
        compute mean along specified axis(es).
        
        args:
            axis: axis or axes along which to compute mean. none means all elements.
            keepdims: if true, keeps reduced dimensions with size 1.
            
        returns:
            tensor: mean result.
        """
        result = np.mean(self.data, axis=axis, keepdims=keepdims)
        return Tensor(result)

    def max(self, axis=None, keepdims=False):
        """
        find maximum value along specified axis(es).
        
        args:
            axis: axis or axes along which to find max. none means all elements.
            keepdims: if true, keeps reduced dimensions with size 1.
            
        returns:
            tensor: maximum value(s).
        """
        result = np.max(self.data, axis=axis, keepdims=keepdims)
        return Tensor(result)

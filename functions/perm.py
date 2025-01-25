import numpy as np


class Function:
    def __init__(self, dimensions: int = 20, beta: float = 1.0) -> None:
        """
        Initialize the function with the given number of dimensions and beta value.

        Args:
            dimensions (int): The number of dimensions (d).
            beta (float): The constant beta in the function formula.
        """
        self.bounds = np.array([(-dimensions, dimensions)] * dimensions)  # n-dimensional bounds
        self.dimensions = dimensions
        self.beta = beta
        self.counter= 0

    def __call__(self, x: np.ndarray = None, **kwargs) -> float:
        """
        Evaluate the function f(x) for a given input vector x.

        Args:
            x (np.ndarray): A 1D numpy array of input values.

        Returns:
            float: The function value.
        """

        if x is not None:
            # Handle input as a numpy array
            if len(x) != self.dimensions:
                raise ValueError(f"Input must have {self.dimensions} dimensions.")
        else:
            # Handle input as keyword arguments
            if len(kwargs) != self.dimensions:
                raise ValueError(f"Input must have {self.dimensions} dimensions.")
            x = np.array([kwargs[f'x{i}'] for i in range(self.dimensions)])

        total_sum = 0.0

        # Outer sum over i
        for i in range(1, self.dimensions + 1):
            inner_sum = 0.0

            # Inner sum over j
            for j in range(1, self.dimensions + 1):
                term = (j ** i + self.beta) * (((x[j - 1] / j) ** i) - 1)
                inner_sum += term

            # Square the inner sum
            total_sum += inner_sum ** 2
        print(-total_sum / (self.dimensions ** 19))

        return -total_sum / (self.dimensions ** 38)

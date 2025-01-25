import numpy as np


class Function:
    def __init__(self, dimensions: int = 100, beta: float = 1.0) -> None:
        """
        Initialize the function with the given number of dimensions and beta value.

        Args:
            dimensions (int): The number of dimensions (d).
        """
        self.bounds = np.array([(-4, 5)] * dimensions)  # n-dimensional bounds
        self.dimensions = dimensions
        self.counter = 0

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

        d = self.dimensions
        total_sum = 0.0

        # Loop over d/4 segments as in the original MATLAB code
        for ii in range(d // 4):
            # Using 0-based indexing in Python (adjusting from MATLAB's 1-based indexing)
            term1 = (x[4 * ii] + 10 * x[4 * ii + 1]) ** 2
            term2 = 5 * (x[4 * ii + 2] - x[4 * ii + 3]) ** 2
            term3 = (x[4 * ii + 1] - 2 * x[4 * ii + 2]) ** 4
            term4 = 10 * (x[4 * ii] - x[4 * ii + 3]) ** 4
            total_sum += (term1 + term2 + term3 + term4) / (10 * self.dimensions ** 2)

        return total_sum


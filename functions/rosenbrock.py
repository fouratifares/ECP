import numpy as np


class Function:
    def __init__(self, dimensions: int = 3) -> None:
        self.bounds = np.array([(-3, 3)] * dimensions)  # n-dimensional bounds
        self.dimensions = dimensions
        self.counter = 0

    def __call__(self, x: np.ndarray = None, **kwargs) -> float:
        """
        Evaluate the Rosenbrock function at a given point using either an array or keyword arguments.

        Args:
            x (np.ndarray, optional): A 1D array of input values (optional).
            **kwargs: Keyword arguments corresponding to the dimensions of the function.

        Returns:
            float: The negative of the Rosenbrock function value.

        Raises:
            ValueError: If the number of input arguments does not match the expected dimensions.
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

        # Generalized Rosenbrock function for n-dimensions
        result = 0
        for i in range(self.dimensions - 1):
            result += (x[i + 1] - x[i] ** 2) ** 2 + (2 - x[i]) ** 2

        return -result / (self.dimensions**2)  # Negate the result to match your original function

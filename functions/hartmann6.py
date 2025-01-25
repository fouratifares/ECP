import numpy as np

class Function:
    def __init__(self, dimensions: int = 6) -> None:
        if dimensions > 6:
            raise ValueError("Dimensions cannot exceed 6 due to constraints of the problem.")
        self.bounds = np.array([(0, 1)] * dimensions)  # n-dimensional bounds
        self.dimensions = dimensions
        self.counter = 0

    def initialize_parameters(self):
        """
        Initialize alpha, A, and P based on the given dimension.
        """
        alpha = np.array([1.0, 1.2, 3.0, 3.2])  # Alpha is always a 4-element vector

        A = np.array([
            [10, 3, 17, 3.50, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14]
        ])

        P = 1e-4 * np.array([
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381]
        ])

        # Return parameters, ensuring we only return the relevant parts for the dimensions
        return alpha, A[:, :self.dimensions], P[:, :self.dimensions]

    def __call__(self, x: np.ndarray = None, **kwargs) -> float:
        """
        Evaluate the function at a given point using either an array or keyword arguments.

        Args:
            x (np.ndarray, optional): A 1D array of input values (optional).
            **kwargs: Keyword arguments corresponding to the dimensions of the function.

        Returns:
            float: The function value.

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

        x = np.array(x)  # Ensure x is a numpy array

        # Initialize parameters based on the dimension
        alpha, A, P = self.initialize_parameters()

        # Compute the summation term for f(x)
        summation = 0.0
        for i in range(len(alpha)):
            # Calculate the exponent term
            exponent = -np.sum(A[i] * (x - P[i]) ** 2)
            summation += alpha[i] * np.exp(exponent)

        return summation

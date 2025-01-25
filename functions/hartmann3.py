import numpy as np


class Function:
    def __init__(self, dimensions: int = 3) -> None:
        self.bounds = np.array([(0, 1)] * dimensions)  # n-dimensional bounds
        self.dimensions = dimensions
        self.counter = 0

    import numpy as np

    def initialize_parameters(self, dim):
        """
        Initialize alpha, A, and P based on the given dimension.
        """
        alpha = np.array([1.0, 1.2, 3.0, 3.2])  # Alpha is always a 4-element vector

        # Create A as a 4x(dim) matrix
        A = np.zeros((4, dim))
        if dim == 3:
            A = np.array([[3.0, 10.0, 30.0],
                          [0.1, 10.0, 35.0],
                          [3.0, 10.0, 30.0],
                          [0.1, 10.0, 35.0]])
        elif dim == 2:
            A = np.array([[3.0, 10.0],
                          [0.1, 10.0],
                          [3.0, 10.0],
                          [0.1, 10.0]])
        elif dim == 1:
            A = np.array([[3.0],
                          [0.1],
                          [3.0],
                          [0.1]])

        # Create P as a 4x(dim) matrix
        P = np.zeros((4, dim))
        if dim == 3:
            P = 10 ** (-4) * np.array([[3689, 1170, 2673],
                                       [4699, 4387, 7470],
                                       [1091, 8732, 5547],
                                       [381, 5743, 8828]])
        elif dim == 2:
            P = 10 ** (-4) * np.array([[3689, 1170],
                                       [4699, 4387],
                                       [1091, 8732],
                                       [381, 5743]])
        elif dim == 1:
            P = 10 ** (-4) * np.array([[3689],
                                       [4699],
                                       [1091],
                                       [381]])

        return alpha, A, P


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

        x = np.array(x)  # Ensure x is a numpy array
        print(x.shape)

        dim = len(x)  # Get the dimensionality of x

        # Initialize parameters based on the dimension
        alpha, A, P = self.initialize_parameters(dim)

        # Compute the summation term for f(x)
        summation = 0
        for i in range(4):
            inner_sum = 0
            for j in range(dim):  # Loop over the dimensions of x
                inner_sum += A[i, j] * (x[j] - P[i, j]) ** 2
            summation += alpha[i] * np.exp(-inner_sum)

        return summation

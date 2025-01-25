import numpy as np


class Function:
    def __init__(self, dimensions: int = 4) -> None:
        self.bounds = np.array([(-10, 10)] * dimensions)  # n-dimensional bounds
        self.dimensions = dimensions
        self.counter = 0


    def __call__(self, x: np.ndarray = None, **kwargs) -> float:
        """
        Evaluate the function at a given point using either an array or keyword arguments.

        Args:
            x (np.ndarray, optional): A 1D array of input values (optional).
            **kwargs: Keyword arguments corresponding to the dimensions of the function.

        Returns:
            float: The value of the function.

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
        # print("x is ", x)
        # Initialize function value
        result = 0

        # Add terms based on the available dimensions
        if self.dimensions >= 1:
            # (x_1 - 1)^2 term
            result += (x[0] - 1) ** 2

        if self.dimensions >= 2:
            # 100(x_1^2 - x_2)^2 + 10.1(x_2 - 1)^2 terms
            result += 100 * (x[0] ** 2 - x[1]) ** 2 + 10.1 * (x[1] - 1) ** 2

        if self.dimensions >= 3:
            # (x_3 - 1)^2 term
            result += (x[2] - 1) ** 2

        if self.dimensions == 4:
            # 90(x_3^2 - x_4)^2 + 10.1(x_4 - 1)^2 + 19.8(x_2 - 1)(x_4 - 1) terms
            result += 90 * (x[2] ** 2 - x[3]) ** 2
            result += 10.1 * (x[3] - 1) ** 2
            result += 19.8 * (x[1] - 1) * (x[3] - 1)

        return -result/10000




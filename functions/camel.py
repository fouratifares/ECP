import numpy as np


class Function:

    def __init__(self) -> None:
        self.bounds = np.array([(-2, 2), (-1, 1)])
        self.optimal_value = 1.0316
        self.optimal_point1 = (0.0898, -0.7126)  # plus/minus
        self.dimensions = 2

    def __call__(self, x: np.ndarray = None, y: np.ndarray = None, **kwargs) -> float:

        if y is not None:
            x = np.array([x, y])

        if x is not None:
            # Handle input as a numpy array
            if len(x) != self.dimensions:
                raise ValueError(f"Input must have {self.dimensions} dimensions.")
        else:
            # Handle input as keyword arguments
            if len(kwargs) != self.dimensions:
                raise ValueError(f"Input must have {self.dimensions} dimensions.")
            x = np.array([kwargs[f'x{i}'] for i in range(self.dimensions)])

        term1 = (4 - 2.1 * x[0] ** 2 + (x[0] ** 4) / 3) * x[0] ** 2
        term2 = x[0] * x[1]
        term3 = (-4 + 4 * x[1] ** 2) * x[1] ** 2

        y = term1 + term2 + term3
        return -y

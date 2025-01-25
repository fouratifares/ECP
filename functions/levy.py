import numpy as np


class Function:

    def __init__(self) -> None:
        self.bounds = np.array([(-10, 10), (-10, 10)])
        self.optimal_value = 0
        self.optimal_point = (1, 1)
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

        term1 = (np.sin(3 * np.pi * x[0])) ** 2
        term2 = (x[0] - 1) ** 2 * (1 + (np.sin(3 * np.pi * x[1])) ** 2)
        term3 = (x[1] - 1) ** 2 * (1 + (np.sin(2 * np.pi * x[1])) ** 2)
        y = term1 + term2 + term3
        return -y

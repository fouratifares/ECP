import numpy as np


class Function:

    def __init__(self) -> None:
        self.bounds = np.array([(-15, 5), (-3, 3)])
        self.optimal_point = (-10, 1)
        self.optimal_value = 0
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

        return -100 * np.sqrt(np.abs(x[1] - 0.01 * (x[0] ** 2))) - 0.01 * np.abs(x[0] + 10)

import numpy as np


class Function:

    def __init__(self) -> None:
        self.bounds = np.array([(-50, 50), (-50, 50)])
        self.optimal_value = 0
        self.optimal_point1 = 0
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

        total = (x[0] ** 2 / 4000) + (x[1] ** 2 / 4000)
        prod = np.cos(x[0] / np.sqrt(1)) * np.cos(x[1] / np.sqrt(2))

        y = total - prod + 1
        return -y

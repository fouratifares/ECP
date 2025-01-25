import numpy as np


class Function:

    def __init__(self) -> None:
        self.bounds = np.array([(-5.12, 5.12), (-5.12, 5.12)])
        self.optimal_value = 186.7309
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

        sum1 = 0
        sum2 = 0

        for i in range(1, 6):
            new1 = i * np.cos((i + 1) * x[0] + i)
            new2 = i * np.cos((i + 1) * x[1] + i)
            sum1 += new1
            sum2 += new2

        y = sum1 * sum2
        return -y/10

import numpy as np


class Function:
    def __init__(self) -> None:
        self.bounds = np.array([(-10, 10), (-10, 10)])
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

        return np.abs(
            np.sin(x[0])
            * np.cos(x[1])
            * np.exp(np.abs(1 - (np.sqrt(x[0] ** 2 + x[1] ** 2)) / np.pi))
        )

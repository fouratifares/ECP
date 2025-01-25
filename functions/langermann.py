import numpy as np


class Function:

    def __init__(self) -> None:
        self.bounds = np.array([(0, 10), (0, 10)])
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

        m = 5
        c = [1, 2, 5, 2, 3]
        A = np.array([[3, 5], [5, 2], [2, 1], [1, 4], [7, 9]])
        d = 2

        outer = 0
        for i in range(m):
            inner = 0
            for j in range(d):
                xj = x[j]
                Aij = A[i, j]
                inner += (xj - Aij) ** 2

            new = c[i] * np.exp(-inner / np.pi) * np.cos(np.pi * inner)
            outer += new

        y = outer
        return -y

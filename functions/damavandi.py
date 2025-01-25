import numpy as np


class Function:

    def __init__(self) -> None:
        self.bounds = np.array([(0, 14), (0, 14)])
        self.optimal_value = 0
        self.optimal_point = (2, 2)
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

        # Calculate the numerator and denominator for the sine terms
        sin_term_x = np.sin(np.pi * (x[0] - 2))
        sin_term_y = np.sin(np.pi * (x[1] - 2))
        denominator = np.pi ** 2 * (x[0] - 2) * (x[1] - 2)

        # Handle the case where the denominator is zero
        # with np.errstate(divide='ignore', invalid='ignore'):
        fraction = np.abs(sin_term_x * sin_term_y / denominator) ** 5

        fraction = np.nan_to_num(fraction, nan=1)  # Convert NaNs to 0 (when denominator is zero)

        # Calculate the Damavandi function value
        y = (1 - fraction) * (2 + (x[0] - 7) ** 2 + 2*(x[1] - 7) ** 2)
        return -y

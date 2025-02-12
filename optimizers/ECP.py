"""
Repository: https://github.com/fouratifares/ECP
Paper: https://arxiv.org/pdf/2502.04290v1
Please cite this work if you use this code in your work.
ECP algorithm for global optimization (Every Call is Precious)
"""

from collections import deque
from utils.utils import *
import numpy as np


def ECP(f, n: int, epsilon=1e-2, tau_=1.001, C=1000):
    """
    f: The class of the function to be maximized (class)
    n: The number of function evaluations (int)
    epsilon: A small value (epsilon_1 > 0)
    tau_: A scaling factor (tau_ > 1)
    C: A constant (C > 1)
    """

    # Initialize variables
    t = 1
    tau = max(1 + (1 / (n * f.dimensions)), tau_)

    # Generate the first random point
    X_1 = Uniform(f.bounds)
    nb_samples = 1

    # Track the number of samples in the last step
    last_nb_samples = deque([1], maxlen=2)

    # Initialize the points and corresponding function values
    points = X_1.reshape(1, -1)
    values = np.array([f(X_1)])

    # Store the current epsilon value
    epsilons = [epsilon]

    # Main optimization loop
    while t < n:
        count_inner_growth = 0
        while True:
            # Generate the next random point
            X_tp1 = Uniform(f.bounds)
            nb_samples += 1
            last_nb_samples[-1] = nb_samples

            # Check if the point satisfies the acceptance condition
            if acceptance_condition(X_tp1, values, epsilon, points, strict=False):
                points = np.concatenate((points, X_tp1.reshape(1, -1)))
                break

            # Check if the growth condition is met
            elif growth_condition(last_nb_samples, C):
                count_inner_growth += 1
                epsilon *= tau
                last_nb_samples[-1] = 0

        # Evaluate the function at the new point
        value = f(X_tp1)
        t += 1
        epsilon *= tau
        epsilons.append(epsilon)

        # Add the new point and its value to the results
        values = np.concatenate((values, np.array([value])))

        # Reset the sample count for the next iteration
        last_nb_samples.append(0)

    return points, values, np.array(epsilons)

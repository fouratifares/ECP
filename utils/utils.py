import numpy as np


def Uniform(X: np.array):
    """
    Generates a random point within the feasible region X.

    X: The feasible region (numpy array).
    """

    theta = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        theta[i] = np.random.uniform(X[i, 0], X[i, 1])
    return theta


def growth_condition(last_nb_samples, max_slope):
    """
    Checks whether the slope of the last points of the nb_samples vs. nb_evaluations curve exceeds max_slope.
    """
    slope = (last_nb_samples[-1] - last_nb_samples[0]) / (len(last_nb_samples) - 1)
    return slope > max_slope


def acceptance_condition(x, values, epsilon, points, strict=False):
    """
    Verifies the acceptance condition based on the current values and explored points.

    values: Array of function values from previously explored points (numpy array).
    x: The point to be evaluated (numpy array).
    epsilon: A small value used for the condition (float).
    points: The set of previously explored points (numpy array).
    strict: Whether the condition should be strictly greater (bool).
    """
    max_val = np.max(values)
    left_min = np.min(values + epsilon * np.linalg.norm(x - points, ord=2, axis=1))

    if strict:
        return left_min > max_val
    else:
        return left_min >= max_val

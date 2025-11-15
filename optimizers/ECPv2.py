from collections import deque
from utils.utils import *
import numpy as np
import time
import os


def growth_condition(last_nb_samples, max_slope):
    """
    Check if the slope of the last points of the nb_samples vs nb_evaluations curve
    is greater than max_slope.
    """
    slope = (last_nb_samples[-1] - last_nb_samples[0]) / (len(last_nb_samples) - 1)
    return slope > max_slope


def acceptance_condition_(x, values, eps, points, max, strict=False):
    """
    values: set of values of the function we explored (numpy array)
    x: point to check (numpy array)
    points: set of points we have explored (numpy array)
    """
    max_val = max
    left_min = np.min(
        values.reshape(-1) + eps * np.linalg.norm(x - points, ord=2, axis=1)
    )
    if strict:
        return left_min > max_val
    else:
        return left_min >= max_val


class FixedProjection:
    def __init__(self, input_dim, target_dim, random_state=42):
        """
        input_dim: Original number of dimensions
        target_dim: Target reduced dimensions
        """
        rng = np.random.default_rng(seed=random_state)
        # Generate a fixed Gaussian projection matrix
        self.proj_matrix = rng.normal(loc=0.0, scale=1.0, size=(input_dim, target_dim)) / np.sqrt(target_dim)

    def transform(self, X):
        """Project data using the fixed random matrix"""
        return X @ self.proj_matrix


def ECPv2(f, n: int,
          epsilon=1e-2, tau_=1.001, c=1e3,
          m=10, delta=0.667, beta=0, projection_dim='auto', lower_bound_epsilon=True,
          verbose=False):
    """
    f: The class of the function to be maximized (class)
    n: The number of function evaluations (int)
    epsilon: A small value (epsilon_1 > 0)
    tau_: A scaling factor (tau_ > 1)
    c: A constant (c > 1)
    delta: Projection error (if 0 then no projection at all)
    lower_bound_epsilon: if true then use the lower-bounding technique of ECPv2
    m: Number of worst-performing points used in acceptance check
    projection_dim: 'auto' or some preferred dimension d' < d
    """

    # Initialize variables
    t = 1
    tau = max(1 + (1 / (n * f.dimensions)), tau_)
    diam = np.linalg.norm(np.array([b[1] - b[0] for b in f.bounds]))
    # m = max(1, int(m * np.log(n)))
    m = max(1, m)

    # report times
    times = []
    best_so_far = []
    start_time = time.time()

    # Generate the first random point
    X_1 = Uniform(f.bounds)
    nb_samples = 1

    # Track the number of samples in the last step
    last_nb_samples = deque([1], maxlen=2)

    #  Initialize the Random Projection Matrix if delta > 0 (and n > 500)
    # if n * delta > 10:
    if delta > 0:

        if projection_dim == "auto":
            lower_dim = int(8 * np.log(beta * n) / (delta ** 2 - delta ** 3))
        else:
            lower_dim = projection_dim

        if verbose:
            print('lower_dim:', lower_dim)

        # Only project if the dimensionality is indeed large
        if lower_dim < f.dimensions:
            PR = FixedProjection(input_dim=f.dimensions, target_dim=lower_dim)
            if verbose:
                print(f'moving to lower-dimension {lower_dim}')
        else:
            if verbose:
                print(f'not moving to a lower-dimension')
            delta = 0
    else:
        delta = 0

    # Initialize the points and corresponding function values
    points = X_1.reshape(1, -1)
    values = np.array([f(X_1)])

    f_max_t = values[0]
    f_min_t = values[0]

    # Only project if delta > 0
    if delta > 0:
        points = PR.transform(points)

    worst_points = points.copy()
    worst_values = values.copy()

    # Store the current epsilon value
    if lower_bound_epsilon:
        if verbose:
            print(f'initial epsilon {epsilon}')
            print(f'LB epsilon {(f_max_t - f_min_t) / diam}')
        epsilon = max(epsilon, (f_max_t - f_min_t) / diam)
    epsilons = [epsilon]

    # Main optimization loop
    while t < n:
        count_inner_growth = 0

        # Only subset if m < n; Otherwise revert to ECP
        considered_points = worst_points  # if m < t else points
        considered_values = worst_values  # if m < t else values

        if m < len(values) and verbose:
            print(f'm technique is being used with m {m}')

        while True:
            # Generate the next random point
            X_tp1 = Uniform(f.bounds)
            nb_samples += 1
            last_nb_samples[-1] = nb_samples

            # Only project if delta > 0
            if delta > 0:
                X_projected = PR.transform(X_tp1)
            else:
                X_projected = X_tp1

            # Check if the point satisfies the acceptance condition
            scaled_epsilon = epsilon / np.sqrt(1 - delta)
            if acceptance_condition_(X_projected, considered_values, scaled_epsilon, considered_points, f_max_t):
                points = np.concatenate((points, X_projected.reshape(1, -1)))
                break

            # Check if the growth condition is met
            elif growth_condition(last_nb_samples, c):
                count_inner_growth += 1
                epsilon *= tau
                last_nb_samples[-1] = 0

        # Evaluate the function at the new point
        value = f(X_tp1)
        values = np.concatenate((values, np.array([value])))

        # Determine acceptance set from k worst points
        if m < len(values):
            idxs = np.argsort(values)[:m]
            worst_values = values[idxs]
            worst_points = points[idxs]
        else:
            worst_values = values
            worst_points = points

        f_max_t = max(f_max_t, value)
        f_min_t = min(f_min_t, value)
        elapsed = time.time() - start_time
        times.append(elapsed)
        best_so_far.append(f_max_t)

        if verbose:
            print(times)
            print(best_so_far)

        t += 1

        # NEW: Lower-bound epsilon update after evaluation
        if lower_bound_epsilon:
            if verbose:
                print(f'current epsilon {epsilon}')
                print(f'LB epsilon {(f_max_t - f_min_t) / diam}')
            epsilon = max(epsilon * tau, (f_max_t - f_min_t) / diam)
        else:
            epsilon *= tau
        epsilons.append(epsilon)

        # Reset the sample count for the next iteration
        last_nb_samples.append(0)

    # Write wall-time and best-so-far to text file
    os.makedirs("results/times", exist_ok=True)

    with open("results/times/ECPv2_time.py", "w") as f_out:
        f_out.write(f"times = {times}\n")
        f_out.write(f"best_so_far = {best_so_far}\n")

    # return points, values, times, best_so_far
    return points, values, np.array(epsilons)

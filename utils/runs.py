import argparse
import os
import numpy as np
import time
from optimizers.ECP import ECP


def convert_to_function(name):
    """
    Converts a function name (string) into the actual function reference.

    Args:
    - name: The name of the function as a string.

    Returns:
    - A list containing the corresponding function reference.
    """

    functions = []

    if name in globals():
        functions.append(globals()[name])
    else:
        raise ValueError(f"'{name}' is not defined or imported.")

    return functions


def cli():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--function", "-f", type=str, help="Function to maximize", required=True
    )
    parser.add_argument(
        "--n_eval", "-n", type=int, help="Number of evaluations (budget)", required=True
    )
    parser.add_argument("--n_run", "-r", type=int, help="Number of runs", required=True)
    parser.add_argument("--optimizers_fct", "-o", type=str, help="List of optimizers", required=False, default=None)
    return parser.parse_args()


def runs(
        n_run: int,
        n_eval: int,
        f,
        optimizer,
        method: str,
        args,
):
    """
    Runs the optimizer multiple times and returns the results of the final run.

    Args:
    - n_run: Number of runs (int)
    - n_eval: Number of function evaluations per run (int)
    - f: The function class to optimize (class)
    - optimizer: The optimizer function to use (function)
    - method: The optimizer method name (str)
    - args: Command-line arguments (Namespace)

    Returns:
    - The points, values, epsilon values, and best result summary.
    """
    print(f"Using Method: {method}")

    # Initialize result tracking for each evaluation step
    chunk = 1
    chunk_results = {i: [] for i in range(chunk, n_eval + 1, chunk)}

    times, values, num_evals, all_values = [], [], [], []

    for i in range(n_run):
        start_time = time.time()
        points, run_values, epsilons = optimizer(f, n=n_eval)
        all_values.append(run_values)

        times.append(time.time() - start_time)

        # Track the best values for each evaluation chunk
        for i in range(chunk, n_eval + 1, chunk):
            chunk_values = run_values[:i]
            if chunk_values.size > 0:
                chunk_results[i].append(np.max(chunk_values))

        num_evals.append(len(run_values))
        if len(run_values) > 0:
            values.append(np.max(run_values))

    # Write results to file
    with open(f"results/results_{method}.txt", "a") as file:
        file.write(f"Method: {method}\n")
        for i in range(chunk, n_eval + 1, chunk):
            avg = np.mean(chunk_results[i])
            std = np.std(chunk_results[i])
            file.write(f"Avg of best max until {i:.2f}: {avg:.5f}, std: {std:.5f}\n")
            if i == 50:
                file.write(f"Avg of best max until {i:.2f}: {avg:.2f} ({std:.2f})\n")
        file.write(f"Avg number of final function evaluations: {np.mean(num_evals):.2f}\n")
        file.write(f"Avg best maximum: {np.mean(values):.2f} ({np.std(values):.2f})\n\n")

        best_r = f"{np.mean(values):.2f} ({np.std(values):.2f}) &"

    return points, values, epsilons, best_r


def directories(args):
    """Creates the results directory if it doesn't exist."""
    if not os.path.exists("results/"):
        os.makedirs("results/")

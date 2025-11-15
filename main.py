import importlib
import random
import sys
import numpy as np
from optimizers.ECP import ECP
from optimizers.ECPv2 import ECPv2
from utils.runs import cli, directories, runs, convert_to_function

# Adding the path for additional functions
sys.path.append("./functions")

if __name__ == "__main__":

    # Parse command line arguments
    args = cli()

    # Set the optimizer function if provided, else default to ECP
    if args.optimizers_fct is None:
        optimizers_fct = [ECP]
    else:
        optimizers_fct = convert_to_function(args.optimizers_fct)

    # Set random seed for consistency across runs
    np.random.seed(42)
    random.seed(42)

    # Create necessary directories
    directories(args)

    # Adjust the function argument to extract the function name
    if len(args.function.split("/")) > 1:
        args.function = args.function.split("/")[1]
    # Remove the file extension (.py)
    args.function = args.function.split(".")[0]

    # Log the objective function in a results file
    with open(f"results/results_{optimizers_fct[0].__name__}.txt", "a") as file:
        file.write(f"objective: {args.function}\n\n")

    # Dynamically load the specified function
    function_module = importlib.import_module(args.function)
    function_instance = function_module.Function()

    # Run the optimization process multiple times for each optimizer
    optimizer_names = [optimizer.__name__ for optimizer in optimizers_fct]
    for i, optimizer_name in enumerate(optimizer_names):
        points, values, epsilons, best_r = runs(
            args.n_run,
            args.n_eval,
            function_instance,
            optimizers_fct[i],
            optimizer_name,
            args,
        )

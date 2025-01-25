# ECP Algorithm for Global Optimization

This repository implements **Every Call is Precious (ECP)** algorithm for solving non-convex (black-box) global optimization problems, with unkown Lipschitz constants, as introduced in our AISTATS paper. 

![plot](figures/surface_plot.png)

## Getting Started

Follow the instructions below to run the ECP algorithm and reproduce the results in our paper.

### Prerequisites

Ensure you have the necessary dependencies installed. You can install them using pip by running:

```bash
pip install -r requirements.txt
```

### Running the Algorithm
You can run the ECP algorithm with the following command:

```bash
python main.py --function <path_to_function_class> -n <num_function_evaluations> -r <num_repetitions> -o ECP
```

#### Arguments:

```--function <path_to_function_class>```: The function to optimize, specified either by its name (e.g., ```ackley```, ```rastrigin```) or by providing a path to the function class.

```-n <num_function_evaluations>```: The total number of function evaluations to perform during the optimization process.

```-r <num_repetitions>```: The number of repetitions to run the optimization, useful for averaging results.

```-o <optimization_method>```: Specifies the optimization method to use. For this implementation, set this argument to ECP.

### Example Usage
To run the ECP algorithm on the Ackley function with 50 function evaluations and 100 repetitions, use:

```
python main.py --function ackley -n 50 -r 100 -o ECP
```

### Output

After the optimization run completes, the results will be saved in the results/ directory in a file named results_ECP.txt. 


### Citation
If you use this code in your research, please cite the following paper:

@inproceedings{fourati2025ecp,
  title={Every Call is Precious: Global Optimization of Black-Box Functions with Unknown Lipschitz Constants},
  author={Fares Fourati, Salma Kharrat, Vaneet Aggarwal, Mohamed-Slim Alouini},
  booktitle={Proceedings of AISTATS},
  year={2025},
}

# Every Call is Precious: Global Optimization of Black-Box Functions with Unknown Lipschitz Constants

This repository implements **ECP** algorithm for solving non-convex black-box global optimization problems, as introduced in [Every Call is Precious: Global Optimization of Black-Box Functions with Unknown Lipschitz Constants](https://arxiv.org/abs/2502.04290v1). 

<img src="figures/surface_plot.png" width="800" height="300"/>

## Highlights

ECP is a global optimization algorithm for maximization that **minimizes unpromising evaluations** by concentrating on potentially optimal regions. It eliminates the need for estimating the Lipschitz constant, thus avoiding unnecessary evaluations. 

ECP guarantees no-regret performance and achieves **minimax-optimal** regret within finite budgets. 

Empirical results show that ECP **outperforms 10 benchmark algorithms** including Lipschitz, Bayesian, bandits, and evolutionary methods across **30 multi-dimensional optimization problems**, which are available under ```functions/```.

## Quick Tutorials with Google Colab

The following tutorials are available as Jupyter notebooks and can be opened directly in Google Colab for an interactive experience:

1. [Tutorial 1: ECP for Global Optimization of the Available Functions](https://colab.research.google.com/github/fouratifares/ECP/blob/main/notebooks/Tutorial_1_ECP_for_Global_Optimization_of_Available_Functions.ipynb)

2. [Tutorial 2: ECP for Global Optimization of a Custom Function](https://colab.research.google.com/github/fouratifares/ECP/blob/main/notebooks/Tutorial_2_ECP_for_Global_Optimization_of_Custom_Function.ipynb)

3. [Tutorial 3: ECP for Hyperparameter Optimization (HPO)](https://colab.research.google.com/github/fouratifares/ECP/blob/main/notebooks/Tutorial_3_ECP_for_Hyperparameter_Optimization.ipynb)

4. [Tutorial 4: ECP for Global Optimization of a Discrete Function](https://colab.research.google.com/github/fouratifares/ECP/blob/main/notebooks/Tutorial_4_ECP_for_Global_Optimization_of_a_Discrete_Function.ipynb)

## Getting Started

Follow the instructions below to run the ECP algorithm and reproduce the results in [the paper](https://arxiv.org/pdf/2502.04290v1).

#### Install

```bash
git clone https://github.com/fouratifares/ECP.git
```

#### Running the Algorithm

You can run the ECP algorithm with the following command:

```bash
python main.py --function <path_to_function_class> -n <num_function_evaluations> -r <num_repetitions> -o ECP
```

#### Arguments:

```--function <path_to_function_class>```: The function to optimize, specified either by its name (e.g., ```ackley```, ```rastrigin```) or by providing a path to the function class.

```-n <num_function_evaluations>```: The total number of function evaluations to perform during the optimization process.

```-r <num_repetitions>```: The number of repetitions to run the optimization, useful for averaging results.

```-o <optimization_method>```: Specifies the optimization method to use. For this implementation, set this argument to ECP.

#### Example Usage

To run the ECP algorithm on the Ackley function with 50 function evaluations and 100 repetitions, use:

```
python main.py --function ackley -n 50 -r 100 -o ECP
```

#### Outputs and Directories

After the optimization run completes, the results will be saved in the ```results/``` directory in a file named results_ECP.txt. 

```functions/```  Includes implementations of 30 multi-dimensional non-convex optimization problems.

```optimizers/``` Includes implementation of the ECP algorithm

## Implement Custom Objective Function

The Function class defines the structure for a callable function that evaluates a reward based on an input array. It is initialized with bounds and dimensions, and it ensures that the input provided meets the expected dimensionality.

```bash
class Function:

    def __init__(self) -> None:
        # Initialize bounds and dimensions for the function.
        self.bounds = ...  # Define the bounds of the function's domain
        self.dimensions = ...  # Define the dimensionality of the input space

    def __call__(self, x: np.ndarray = None) -> float:
        # Ensure input is a numpy array and has the correct number of dimensions
        if x is not None:
            if len(x) != self.dimensions:
                raise ValueError(f"Input must have {self.dimensions} dimensions.")
        
        # Evaluate the function and return a reward (or some output value)
        reward = ...  # Your function's logic to calculate reward

        return reward
```

## Citation
If you use this code in your research, please cite the following papers:

```
@InProceedings{fourati25ecp,
  title = 	 {Every Call is Precious: Global Optimization of Black-Box Functions with Unknown Lipschitz Constants},
  author =       {Fourati, Fares and Kharrat, Salma and Aggarwal, Vaneet and Alouini, Mohamed-Slim},
  booktitle = 	 {Proceedings of The 28th International Conference on Artificial Intelligence and Statistics},
  pages = 	 {5176--5184},
  year = 	 {2025},
  volume = 	 {258},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {03--05 May},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v258/main/assets/fourati25a/fourati25a.pdf},
  url = 	 {https://proceedings.mlr.press/v258/fourati25a.html},
}
```
```
@article{fourati2025ecpv2,
  title={ECPv2: Fast, Efficient, and Scalable Global Optimization of Lipschitz Functions},
  author={Fourati, Fares and Alouini, Mohamed-Slim and Aggarwal, Vaneet},
  journal={arXiv preprint arXiv:2511.16575},
  year={2025}
}
```

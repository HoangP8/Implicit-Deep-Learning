<div align="center">
  <h1>Implicit Deep Learning Package</h1>
</div>

<p align="center">
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License" height="20" style="border: none;">
  </a>
  <a href="https://implicit-deep-learning.readthedocs.io/en/latest/">
    <img src="https://img.shields.io/badge/documentation-latest-yellow.svg" alt="Documentation" height="20" style="border: none;">
  </a>
  <a href="https://pypi.org/project/torchidl/">
    <img src="https://img.shields.io/pypi/v/torchidl?color=brightgreen&label=PyPI" alt="PyPI" height="20" style="border: none;">
  </a>
</p>


<p align="center">
  <a href="https://implicit-deep-learning.readthedocs.io/en/latest/"><b>Documentation</b></a> 
  •
  <a href="https://github.com/HoangP8/Implicit-Deep-Learning/blob/main/tutorial.ipynb"><b>Quick Tutorial</b></a> 
  •
  <a href="https://github.com/HoangP8/Implicit-Deep-Learning?tab=readme-ov-file#installation"><b>Installation</b></a> 
  •
  <a href="https://github.com/HoangP8/Implicit-Deep-Learning/tree/main/examples"><b>Examples</b></a>
  • 
  <a href="https://github.com/HoangP8/Implicit-Deep-Learning?tab=readme-ov-file#citation"><b>Citation</b></a>
</p>


**Authors**: Hoang Phan, [Bao Tran](https://www.linkedin.com/in/bao-tq/), Chi Nguyen, Bao Truong, Thanh Tran, [Khai Nguyen](https://xkhainguyen.github.io/), [Alicia Y. Tsai](https://www.aliciatsai.com/), [Hong Chu](https://sites.google.com/view/hongtmchu), [Laurent El Ghaoui](https://people.eecs.berkeley.edu/~elghaoui/)

## Introduction
Traditional feedforward neural networks compute hidden states by passing inputs sequentially through layers. In contrast, Implicit Deep Learning determines these hidden states by solving fixed-point equations. This approach offers theoretical advantages in stability, memory efficiency, and expressivity.

<p align="center">
  <img src="docs/assets/implicit.jpg" alt="implicit_figure" width="100%"><br>
</p>

Formally, given input $u \in \mathbb{R}^p$ and output $y \in \mathbb{R}^q$, the implicit model (Figure b) computes the equilibrium hidden state $x \in \mathbb{R^n}$ by solving:

$$
\begin{cases}
\begin{aligned}
x &= \phi(A x + B u), \quad &&\text{(Equilibrium equation)} \\
\hat{y} &= C x + D u, \quad &&\text{(Prediction equation)}
\end{aligned}
\end{cases}
$$

where $\phi$ is a non-linear activation function that satisfies the well-posedness condition (e.g., ReLU), and $A \in \mathbb{R}^{n \times n}, B \in \mathbb{R}^{n \times p}, C \in \mathbb{R}^{q \times n}, D \in \mathbb{R}^{q \times p}$ are learnable parameters.

This formulation can represent any feedforward network. For example, a simple MLP with 2 hidden layers (Figure a):

$$
x_1 = \phi(W_0u) \longrightarrow x_2=\phi(W_1x_1) \longrightarrow \hat{y}(u) = W_2 x_2.
$$

can be rewritten in matrix form as:

$$x = \phi(A x + B u) = \phi \left( \left[\begin{array}{cc} 
0 & W_1 \\
0 & 0 
\end{array}\right] \left[\begin{array}{c} 
x_2 \\
x_1 
\end{array}\right] + \left[\begin{array}{c} 
0 \\
W_0 
\end{array}\right] u \right)
= \phi \left[\begin{array}{c} 
W_1 x_1 \\
W_0 u 
\end{array}\right] = \left[\begin{array}{c} 
x_2 \\
x_1 
\end{array}\right]
$$



$$\hat{y} = C x + D u = \left[\begin{array}{cc} 
W_2 & 0 
\end{array}\right] \left[\begin{array}{c} 
x_2 \\
x_1 
\end{array}\right] + \left[\begin{array}{c} 
0 
\end{array}\right] u = W_2 x_2$$

For a conceptual introduction to Implicit Models, see this article on [Medium](https://medium.com/analytics-vidhya/what-is-implicit-deep-learning-9d94c67ec7b4). For mathematical foundations and technical details, refer to the [SIAM journal paper](https://epubs.siam.org/doi/abs/10.1137/20M1358517).

## Installation
- Via `pip`:
  ```
  pip install torchidl
  ```
- From source:
  ```
  git clone https://github.com/HoangP8/Implicit-Deep-Learning && cd Implicit-Deep-Learning
  pip install -e .
  ```

## Quick Tour
The `idl` package provides a comprehensive framework for implementing and experimenting with implicit models. It includes the foundational `ImplicitModel`, the recurrent variant `ImplicitRNN`, and the state-driven training approach `SIM`. Below are simple examples of how to use each model. More details of the package including the theory and functions can be found in the [documentation](https://implicit-deep-learning.readthedocs.io/en/latest/).

### Example: `ImplicitModel`

```python
from idl import ImplicitModel

# Prepare your dataset
train_loader, test_loader = ...

# Initialize the model
model = ImplicitModel(
    hidden_dim=100,  # Hidden state dimension
    input_dim=3072,  # Input dimension
    output_dim=10,   # Output dimension
).to(device)

# Standard PyTorch training loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(num_epochs): 
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets) 
        loss.backward()  
        optimizer.step()
```

### Example: `SIM` (State-driven Implicit Modeling)

State-driven Implicit Modeling (SIM) is a training method for Implicit Models that uses a pretrained network to synthesize training data. SIM formulates the training process of Implicit Models as a convex optimization problem so that it can be solved stably and efficiently. 

Requirements:
- A pretrained network
- A convex optimization solver

For more details, see the [documentation](https://implicit-deep-learning.readthedocs.io/en/latest/api/sim.html).

```python
import torch
from idl.sim import SIM
from idl.sim.solvers import CVXSolver

# Prepare your dataset
train_loader, test_loader = ...

# Load a pretrained explicit model
explicit_model = MLP(input_dim, hidden_dim, output_dim).to(device)
explicit_model.load_state_dict(torch.load('checkpoint.pt'))

# Initialize SIM model and solver
sim = SIM(activation_fn=torch.nn.functional.relu, device=device)
solver = CVXSolver()

# Train and evaluate
sim.train(solver=solver, model=explicit_model, dataloader=train_loader)
sim.evaluate(test_loader)
```

The package provides multiple solvers including:
- `CVXSolver`: Convex optimization solver
- `ADMMSolver`: Standard ADMM implementation
- `ADMMMultiGPUSolver`: Distributed training across multiple GPUs
- `ProjectedGDLowRankSolver`: Projected gradient descent with low-rank constraints
- `LeastSquareSolver`: Efficient least squares solver using `numpy.linalg.lstsq` (ignores well-posedness constraint)

Besides, custom solvers can also be implemented by extending the `BaseSolver` class.

## Documentation and Examples
More details of the package are provided in the [documentation](https://implicit-deep-learning.readthedocs.io/en/latest/). For a comprehensive introduction, see our [Notebook tutorial](https://github.com/HoangP8/Implicit-Deep-Learning/blob/main/tutorial.ipynb). The [examples](https://github.com/HoangP8/Implicit-Deep-Learning/tree/main/examples) directory contains implementation examples for each model variant.

To run an example:
```
bash examples/idl/idl.sh
```

## Related Works 

**Implicit Deep Learning** \
[SIAM, 2020](https://epubs.siam.org/doi/abs/10.1137/20M1358517) \
Laurent El Ghaoui, Fangda Gu, Bertrand Travacca, Armin Askari, Alicia Y. Tsai

**Constrained Implicit Learning Framework for Neural Network Sparsification** \
[ACML, 2025](https://proceedings.mlr.press/v260/tsai25a.html) \
Alicia Y. Tsai, Wenzhi Gao, Laurent El Ghaoui

**The Extrapolation Power of Implicit Models** \
[IJCAI, 2024](https://arxiv.org/abs/2407.14430) \
Juliette Decugis, Alicia Y Tsai, Max Emerling, Ashwin Ganesh, Laurent El Ghaoui

**State-driven Implicit Modeling for Sparsity and Robustness in Neural Networks** \
[arxiv, 2022](https://arxiv.org/abs/2209.09389) \
Alicia Y. Tsai, Juliette Decugis, Laurent El Ghaoui, Alper Atamtürk

**Implicit Learning in Deep Models: Enhancing Extrapolation Power and Sparsity** \
[Dissertation, 2024](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2024/EECS-2024-214.pdf) \
Alicia Y. Tsai


## Citation

```
@misc{torchidl,
    author = {Hoang Phan and Bao Tran and Chi Nguyen and Bao Truong and Thanh Tran and Khai Nguyen and Alicia Y. Tsai and Hong Chu and Laurent El Ghaoui},
    title = {Implicit Deep Learning Pytorch Package},
    year = {2025},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/HoangP8/Implicit-Deep-Learning}},
}
```

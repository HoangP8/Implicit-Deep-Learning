<div align="center">
  <h1>Implicit Deep Learning Package</h1>
</div>

<p align="center">
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License" height="20" style="border: none;">
  </a>
  <a href="https://pypi.org/project/torchcam/">
    <img src="https://img.shields.io/pypi/v/torchcam.svg?logo=PyPI&logoColor=fff&style=flat-square&label=PyPI" alt="PyPi Version" style="border: none;">
  </a>
  <a href="https://colab.research.google.com/github/frgfm/notebooks/blob/main/torch-cam/quicktour.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Tutorial" style="border: none;">
  </a>
  <a href="https://frgfm.github.io/torch-cam">
    <img src="https://img.shields.io/github/actions/workflow/status/frgfm/torch-cam/page-build.yml?branch=main&label=Documentation&logo=read-the-docs&logoColor=white&style=flat-square" alt="Documentation" style="border: none;">
  </a>
</p>

<p align="center">
  <a href="https://github.com/HoangP8/Implicit-Deep-Learning?tab=readme-ov-file#introduction"><b>Introduction</b></a> 
  • 
  <a href="https://github.com/HoangP8/Implicit-Deep-Learning?tab=readme-ov-file#installation"><b>Installation</b></a> 
  •
  <a href="https://torchdeq.readthedocs.io/en/latest/deq-zoo/model.html"><b>Quick Tour</b></a>
  •
  <a href="https://github.com/HoangP8/Implicit-Deep-Learning?tab=readme-ov-file#contribution"><b>Contribution</b></a> 
  • 
  <a href="https://github.com/HoangP8/Implicit-Deep-Learning?tab=readme-ov-file#contribution"><b>Citation</b></a>
</p>


## Introduction
Implicit Deep Learning finds a hidden state $X$ by solving a fixed-point equation instead of explicitly stacking layers conventionally. Given a dataset with input matrix $U \in \mathbb{R}^{p\times m}$ and output matrix $Y \in \mathbb{R}^{q\times m}$, where each column represents an input or output vector and $m$ is the batch size, the implicit model uses the following equations:

1. State equation:

$$X = \phi (AX + BU),$$

2. Prediction equation:

$$\hat{Y}(U) = CX + DU,$$

where $\phi: \mathbb{R}^{n\times m} \to \mathbb{R}^{n\times m}$ is a nonlinear activation that is strictly increasing and component-wise non-expansive, such as ReLU, tanh or sigmoid. Matrices $A\in \mathbb{R}^{n\times n}$, $B\in \mathbb{R}^{n\times p}$, $C\in \mathbb{R}^{q\times n}$ and $D\in \mathbb{R}^{q\times p}$ are model parameters.

For illustration, below is an implicit model equivalent to a 2-layer feedforward neural network.
<p align="center">
  <img src="figures/ff-illus.jpg" alt="2-layer feedforward" width="80%"><br>
</p>

As opposed to the above figure, the typical implicit model does not have a clear hierarchical, layered structure.
<p align="center">
  <img src="figures/im-illus.jpg" alt="typical implicit model" width="80%"><br>
</p>


To dive deeper into the motivation behind Implicit Models, check out this beginner-friendly article on [Medium](https://medium.com/analytics-vidhya/what-is-implicit-deep-learning-9d94c67ec7b4). If you're curious about the math and technical details, the full framework is explained in [this journal article](https://epubs.siam.org/doi/abs/10.1137/20M1358517).


## Installation
- Install required packages by running:
  ```
  pip install -r requirements.txt
  ```
- Through `pip`:
  ```
  pip install idl
  ```
- From source:
  ```
  git clone https://github.com/HoangP8/Implicit-Deep-Learning && cd Implicit-Deep-Learning
  pip install -e .
  ```

## Quick tour
The `idl` package makes it easy to experiment with all variants of implicit models using just a few lines of code. It includes four models: `ImplicitModel`, `ImplicitRNN`, and `SIM`. For a full breakdown of each model’s architecture and hyperparameters, check out the [documentation](link).

### Example: `ImplicitModel`

Here’s how to use `ImplicitModel`, the most basic form of an implicit model:


```python
from idl import ImplicitModel

# Normal data processing
train_loader, test_loader = ...  # Any dataset users use (e.g., CIFAR10, time-series, ...)

# Define the Implicit Model
model = ImplicitModel(
    hidden_dim=100,  # Size of the hidden dimension
    input_dim=3072,  # Input dimension (e.g., 3*32*32 for CIFAR-10)
    output_dim=10,   # Output dimension (e.g., 10 classes for CIFAR-10)
).to(device)

# Normal training loop
optimizer = ...  # Choose optimizer (e.g., Adam, SGD)
loss_fn = ...    # Choose loss function (e.g., Cross-Entropy, MSE)

for _ in range(epoch): 
    ...
    optimizer.zero_grad()
    loss = loss_fn(model(inputs), targets) 
    loss.backward()  
    optimizer.step()  
    ...
```

For `ImplicitRNN`, the interface is similar to `ImplicitModel`.

### Example: `SIM`

```python
from idl.sim import SIM
from idl.sim.solvers import CVXSolver

# Normal data processing
train_loader, test_loader = ...  # Any dataset users use (e.g., CIFAR10, time-series, ...)

# Define an explicit model
model = MLP(input_dim, hidden_dim, output_dim).to(device)

# Define SIM model
sim = SIM(device=device)

# Loss function and optimizer
optimizer = ...  # Choose optimizer (e.g., Adam, SGD)
loss_fn = ...    # Choose loss function (e.g., Cross-Entropy)

# Train and evaluate the SIM model
sim.train(solver=CVXSolver(), model=explicit_model, dataloader=train_loader)
sim.evaluate(test_loader)
```

We provide several solvers for `SIM`, including `ADMMSolver`, `ADMMMultiGPUSolver`, `CVXSolver`, `ProjectedGDLowRankSolver`, `LeastSquareSolver`.  For more information on their hyperparameters, please refer to the [documentation](link).

### Full instruction
To get familiar with the framework, start with the [Notebook tutorial](link). We also included a folder `example` in the repository [here](link), which contains subfolders for each model. Each subfolder includes a script for easy experimentation. For example, to run the IDL example, adjust the parameters in the script and execute:

  ```
  bash examples/idl/idl.sh
  ```


## Contribution

## Citation

```
@article{el2021implicit,
  title={Implicit deep learning},
  author={El Ghaoui, Laurent and Gu, Fangda and Travacca, Bertrand and Askari, Armin and Tsai, Alicia},
  journal={SIAM Journal on Mathematics of Data Science},
  volume={3},
  number={3},
  pages={930--958},
  year={2021},
  publisher={SIAM}
}

```

```
@article{tsai2022state,
  title={State-driven implicit modeling for sparsity and robustness in neural networks},
  author={Tsai, Alicia Y and Decugis, Juliette and Ghaoui, Laurent El and Atamt{\"u}rk, Alper},
  journal={arXiv preprint arXiv:2209.09389},
  year={2022}
}
```

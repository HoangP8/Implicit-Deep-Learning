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
  <a href="https://torchdeq.readthedocs.io/en/latest/get_started.html"><b>Introduction</b></a> 
  • 
  <a href="https://colab.research.google.com/drive/12HiUnde7qLadeZGGtt7FITnSnbUmJr-I?usp=sharing"><b>Installation</b></a> 
  •
  <a href="https://torchdeq.readthedocs.io/en/latest/deq-zoo/model.html"><b>Quick Tour</b></a>
  •
  <a href="TODO.md"><b>Contribution</b></a> 
  • 
  <a href="README.md#citation"><b>Citation</b></a>
</p>


## Introduction
Implicit Deep Learning finds a hidden state $X$ by solving a fixed-point equation instead of explicitly stacking layers conventionally. Given a dataset with input matrix $U \in \mathbb{R}^{p\times m}$ and output matrix $Y \in \mathbb{R}^{q\times m}$, where each column represents an input or output vector and $m$ is the batch size, the implicit model uses the following equations:

1. State equation:

$$X = \phi (AX + BU),$$

2. Prediction equation:

$$\hat{Y}(U) = CX + DU,$$

where $\phi: \mathbb{R}^{n\times m} \to \mathbb{R}^{n\times m}$ is a nonlinear activation that is strictly increasing and component-wise non-expansive, such as ReLU, tanh or sigmoid. Matrices $A\in \mathbb{R}^{n\times n}$, $B\in \mathbb{R}^{n\times p}$, $C\in \mathbb{R}^{q\times n}$ and $D\in \mathbb{R}^{q\times p}$ are model parameters.

For illustration, below is an implicit model equivalent to a 2-layer feedforward neural network: 
![feedforward-implicit-illus](https://github.com/alicia-tsai/implicit-deep-learning/blob/main/figures/ff-illus.jpg)


As opposed to the above network, the typical implicit model does not have a clear hierachical, layered structure:
![feedforward-implicit-illus](https://github.com/alicia-tsaiL/implicit-deep-learning/blob/main/figures/im-illus.jpg)

Journal article: https://epubs.siam.org/doi/abs/10.1137/20M1358517

Press article: https://medium.com/analytics-vidhya/what-is-implicit-deep-learning-9d94c67ec7b4

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

## Interface

Example for `ImplicitModel` or ``ImplicitRNN``:

```python
from idl import ImplicitModel

# Normal data processing
train_loader, test_loader = ...  # Any dataset users use (e.g., CIFAR10, time-series, ...)

# Define the Implicit Model
model = ImplicitModel(
    hidden_dim=100,  # Size of the hidden dimension
    input_dim=3072,  # Input dimension (e.g., 3*32*32 for CIFAR-10)
    output_dim=10,   # Output dimension (e.g., 10 classes for CIFAR-10)
)

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



Example for `IDLHead`:
```python
from idl import IDLHead

# Load data as normal
train_data, val_data = load_data()

# Define the Transformer Model (e.g., GPT)
model = GPTLanguageModel(
    vocab_size=...,  # Vocabulary size
    n_embd=...,      # Embedding dimension
    block_size=...,  # Context length
    n_layer=...,     # Number of layers
    n_head=...,      # Number of attention heads
    dropout=...      # Dropout rate
)

# Replace standard attention heads with IDL attention heads
for i in range(n_layer):
    model.blocks[i].sa.heads = nn.ModuleList([
        IDLHead(
            n_embd // n_head,  # Dimension per head
            n_embd,            # Embedding dimension
            block_size,        # Context length
        ) 
        for _ in range(args.n_head)
    ])

# Normal GPT-model training
train_model(args, model, train_data, val_data, device, log_file)
```

Note:
- For `ImplicitModel`, `ImplicitRNN`, and `IDLHead`, more examples are provided in the `examples` folder. Each model comes with a `.sh` script for easy execution. This is an example for IDL, please adjust the script parameters as needed and run:
  ```
  bash examples/idl/idl.sh
  ```
- For a full list of hyperparameters and detailed usage, refer to the [`documentation`](https://www.youtube.com/).

## Contribution

## Citation

# Implicit-Deep-Learning
IM + SIM + Attention

## Project Structure

Below is the project's repository structure:

```plaintext 
Project
├─ 📂examples                    
│   ├─ 📂im 
│   ├─ 📂im_rnn
│   ├─ 📂im_attention
│   └─ 📂sim
│
├─ 📂docs  
│
├─ 📂src
│   └─ 📂idl
│   │   ├─ 📃base_function.py
│   │   ├─ 📃ImplicitModel.py
│   │   ├─ 📃ImplicitRNN.py
│   │   └─ 📂sim
│   │   │   ├─ 📃base_function.py
│   │   │   └─ 📃SIM.py
└─ 📃LICENSE
└─ 📃README.md
└─ 📃pyproject.toml
```

## Interface

Here’s a sample usage of the `ImplicitModel` or ``ImplicitRNN`` within our framework:

```python
from .idl.implicit_base_model import ImplicitModel

# Normal data processing
train_loader, test_loader = ...  # Any dataset users use (e.g., CIFAR10, time-series, ...)

# Define the Implicit Model
model = ImplicitModel(hidden_dim=..., input_dim=..., output_dim=...,
                      low_rank=T/F, rank=...,
                      mitr=..., grad_mitr=..., tol=..., grad_tol=...,
                      f=ImplicitFunctionInf)

# Normal training loop
optimizer = ...  # Choose optimizer (e.g., Adam, SGD)
loss_fn = ...    # Choose loss function (e.g., Cross-Entropy, MSE)

for _ in range(epoch): 
    optimizer.zero_grad() 
    loss.backward()  
    optimizer.step()  
    ...
```

- By default, the parameters `mitr=grad_mitr=300`, and `tol=grad_tol=3e-6`.
- The default value of `low_rank` is `False`, meaning the model is full rank by default. Users can easily switch to a low-rank version.
- Users need to define `hidden_dim` for the implicit model. The `input_dim` represents the input dimension vector; similarly for `output_dim`.
- The default function `f=ImplicitFunctionInf` is the wellposedness condition for the infinity norm of matrix A.
- Example CIFAR-10, use `input_dim=3*32*32=3072` for the 32x32 RGB images and `output_dim=10` for 10 classes.
- We want a low-rank Implicit model with `hidden_dim=100`:

```python
model = ImplicitModel(hidden_dim=100, input_dim=3072, output_dim=10, low_rank=True, rank=2)
```
- Indeed users can change any parameters they want, by setting the value of parameters.
- Same approach for `ImplicitRNN` with time-series dataset.

## TODO

- [x] Code SIM core functions.
- [x] Code Implicit core functions.
   - [x] Implicit model + LoRa
   - [x] Implicit RNN model + LoRa
- [x] Code examples for `im` and `im_rnn` 
   - [x] Refactor Time-series data processing + training
   - [x] Test Inference + Debug   
- [x] Refactor `im_attention` example
   - [x] Flexibility in base self-attention and Lipschitz version
   - [x] Refactor the functions effectively
   - [ ] Users can use easily
- [ ] Refactor `sim` example
   - [ ] Get different solvers
   - [ ] Different experiments
- [x] Docstring (Important)
   - [x] Implicit (ver1)
   - [ ] SIM 
- [ ] Debug, check, and test (1 week)


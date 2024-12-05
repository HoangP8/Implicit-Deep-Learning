# Implicit-Deep-Learning
IM + SIM + Attention

## Project Structure

Below is the project's repository structure:

```plaintext 
Project
├─ 📂examples                    
│   ├─ 📂im 
│   │   ├─ 📃data.py (MNIST and CIFAR10)
│   │   └─ 📃main.py 
│   ├─ 📂im_rnn
│   │   ├─ 📃data.py (time-series synthetic dataset)
│   │   └─ 📃main.py 
│   ├─ 📂im_attention
│   │   ├─ 📃data.py
│   │   └─ 📃main.py 
│   └─ 📂SIM
│   │   ├─ 📃data.py
│   │   └─ 📃main.py 
│   
├─ 📂idl
│   ├─ 📃base_function.py
│   ├─ 📃ImplicitModel.py
│   └─ 📃ImplicitRNN.py
│ 
├─ 📂sim
│   ├─ 📃base_function.py
│   └─ 📃SIM.py
│  
└─ 📃introduction.ipynb                    
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


## TODO

- [ ] Code sim core functions.
- [ ] Code im core functions.
- [ ] Infer + Check core functions
- [ ] Merge im core functions with Attention examples
- [ ] Implicit-zoo
   - [ ] im
   - [ ] sim
   - [ ] attention
- [ ] Infer + Check zoo
- [ ] Docstring


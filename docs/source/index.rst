.. idl documentation master file, created by
   sphinx-quickstart on Thu Jan 23 23:56:54 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to IDL Package
======================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api/idl
   api/sim
   api/examples

IDL (Implicit Deep Learning) is a Python package that implements implicit deep learning models (with a specialized recurrent version) and the state-driven training approach.

Key Features
------------

* Implicit Modeling with optional low-rank training
* State-driven Implicit Modeling (SIM) training approach
* Easy to use, seamless integration with Pytorch autograd

Getting Started
===============

Installation
------------

Install using pip:

.. code-block:: bash

   pip install idl

Install from source:

.. code-block:: bash

   git clone https://github.com/HoangP8/Implicit-Deep-Learning
   cd Implicit-Deep-Learning
   pip install -e .

Basic Usage
-----------

Here's a simple example using `ImplicitModel`:

.. code-block:: python

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
   
* Load your dataset and train as usual, the forward and backward passes are fully packaged.
* `ImplicitModel` is simple to use with just a few lines of code.
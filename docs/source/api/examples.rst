Examples
========

MNIST Classification
--------------------

.. code-block:: python

   from idl import ImplicitModel
   import torch
   import torchvision

   # Load MNIST dataset
   train_loader = torch.utils.data.DataLoader(
       torchvision.datasets.MNIST('./data', train=True, download=True),
       batch_size=32
   )

   # Create and train model
   model = ImplicitModel(hidden_dim=100, input_dim=784, output_dim=10)
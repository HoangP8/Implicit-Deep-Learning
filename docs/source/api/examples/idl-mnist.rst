MNIST Classification with Implicit Model
----------------------------------------

The following example shows how to use the :class:`idl.implicit_base_model.ImplicitModel` class to train a simple classification model on the MNIST dataset.

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

   # Define optimizer
   optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

   # Train model
   for epoch in range(10):
       for batch_idx, (data, target) in enumerate(train_loader):
           optimizer.zero_grad()
           output = model(data)
           loss = torch.nn.functional.nll_loss(output, target)
           loss.backward()
           optimizer.step()

   # Save model
   torch.save(model.state_dict(), 'model.pt')

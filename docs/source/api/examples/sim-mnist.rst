MNIST Classification with SIM
-----------------------------

The following example shows how to use the :class:`idl.sim.sim.SIM` class to train a simple classification model on the MNIST dataset.


First, we need to train a simple Feedforward Neural Network on the MNIST dataset.

.. code-block:: python

   import torch
   import torchvision

   # Load MNIST dataset
   train_loader = torch.utils.data.DataLoader(
       torchvision.datasets.MNIST('./data', train=True, download=True),
       batch_size=32
   )
   
   # Define model
   model = torch.nn.Sequential(
       torch.nn.Flatten(),
       torch.nn.Linear(784, 64),
       torch.nn.ReLU(),
       torch.nn.Linear(64, 32),
       torch.nn.ReLU(),
       torch.nn.Linear(32, 10)
   )
   
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

Now, we can use the :class:`idl.sim.sim.SIM` class to train the Implicit Model on the MNIST dataset.

.. code-block:: python

   import torch
   import torchvision
   from idl.sim import SIM
   from idl.sim.solvers import CVXSolver

   # Load MNIST dataset. The dataset is quite large, but we only need a small subset to train our implicit model with the state-driven method.
   train_loader = torch.utils.data.DataLoader(
       torchvision.datasets.MNIST('./data', train=True, download=True),
       batch_size=32
   )
   selected_indices = random.sample(
       range(len(train_loader.dataset)), 2000
   )
   subset = Subset(train_loader.dataset, selected_indices)
   subset_loader = DataLoader(subset, batch_size=1000, shuffle=True)

   # Load pretrained explicit model
   model = torch.nn.Sequential(
       torch.nn.Flatten(),
       torch.nn.Linear(784, 64),
       torch.nn.ReLU(),
       torch.nn.Linear(64, 32),
       torch.nn.ReLU(),
       torch.nn.Linear(32, 10)
   )
   model.load_state_dict(torch.load('model.pt'))

   # Define SIM model
   sim = SIM(activation_fn=torch.nn.functional.relu, device="cuda", dtype=torch.float32)

   # Define solver
   solver = CVXSolver()

   # Train SIM
   sim.train(solver=solver, model=model, dataloader=subset_loader)

Projected GD Solver with Low-rank Regularization
------------------------------------------------

.. autoclass:: torchidl.sim.solvers.projected_gd_lowrank.ProjectedGDLowRankSolver
   :members: solve

Example usage:

.. code-block:: python

   import torch
   from torchidl import SIM
   from torchidl import ProjectedGDLowRankSolver

   # Load dataset
   dataloader = ...

   # Load a pretrained explicit model
   explicit_model = ...
   explicit_model.load_state_dict(torch.load("checkpoint.pt"))

   # Define the SIM model
   sim = SIM(activation_fn=torch.nn.functional.relu, device="cuda", dtype=torch.float32)

   # Define the solver and solve the state-driven training problem
   solver = ProjectedGDLowRankSolver()
   sim.train(solver=solver, model=explicit_model, dataloader=dataloader)
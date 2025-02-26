CVX Solver
----------

**Note**: In our experiments, we use the `MOSEK` solver which is not the default solver of CVXPY. It is powerful, but it requires a license and additional installation.

.. autoclass:: idl.sim.solvers.cvx.CVXSolver
   :members: solve

Example usage:

.. code-block:: python

   import torch
   from idl.sim import SIM
   from idl.sim.solvers import CVXSolver

   # Load dataset
   dataloader = ...

   # Load a pretrained explicit model
   explicit_model = ...
   explicit_model.load_state_dict(torch.load("checkpoint.pt"))

   # Define the SIM model
   sim = SIM(activation_fn=torch.nn.functional.relu, device="cuda", dtype=torch.float32)

   # Define the solver and solve the state-driven training problem
   solver = CVXSolver()
   sim.train(solver=solver, model=explicit_model, dataloader=dataloader)
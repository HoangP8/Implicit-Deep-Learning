ADMM Solvers
------------

.. autoclass:: idl.sim.solvers.admm_consensus.ADMMSolver
   :members: solve
   
.. autoclass:: idl.sim.solvers.admm_consensus_multigpu.ADMMMultiGPUSolver
   :members: solve

Example usage:

.. code-block:: python

   import torch
   from idl.sim import SIM
   from idl.sim.solvers import ADMMSolver

   # Load dataset
   dataloader = ...

   # Load a pretrained explicit model
   explicit_model = ...
   explicit_model.load_state_dict(torch.load("checkpoint.pt"))

   # Define the SIM model
   sim = SIM(activation_fn=torch.nn.functional.relu, device="cuda", dtype=torch.float32)

   # Define the solver and solve the state-driven training problem
   solver = ADMMSolver()
   sim.train(solver=solver, model=explicit_model, dataloader=dataloader)
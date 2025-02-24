SIM (State-driven Implicit Models)
==================================

A standard Implicit Model is modeled by the following equation:

.. math::
   \begin{aligned}
      X &= \phi(A X + B U) \quad &\text{(Equilibrium equation)}, \\
      \hat{Y} &= C X + D U \quad &\text{(Prediction equation)},
   \end{aligned}

The State-driven training method (described in the paper `State-driven Implicit Models <https://arxiv.org/abs/2209.09389>`_) is a method to distill implicit models 
from pre-trained explicit networks by matching the *internal state vectors* of the explicit networks.

Given an input matrix :math:`U \in \mathbb{R}^{p \times m}`, synthetic state matrices :math:`Z \in \mathbb{R}^{n \times m}` (the pre-activation vectors of the explicit networks), 
:math:`X \in \mathbb{R}^{n \times m}` (the post-activation vectors of the explicit networks), and the synthetic output matrix :math:`\hat{Y} \in \mathbb{R}^{q \times m}`, 
the SIM training method aims to solve the following convex optimization problem:

.. math::
   \begin{aligned}
      & \min_{M} \quad f(M)\\
      & \text{s.t.} \quad
      Z = AX + BU, \\
      & \quad \quad \hat{Y} = CX + DU, \\
      & \quad \quad \|A\|_\infty \leq \kappa.
   \end{aligned}

Please refer to the paper for more details about the SIM training method. In order to train the implicicit model, we need to:

1. Define a SIM model that extracts the state vectors and initializes the problem.
2. Create a solver to solve the convex optimization problem.

For the first phase, we abstracted all the necessary components into the :class:`idl.sim.sim.SIM` class.

For the second phase, several solvers are already provided in the next sections. 
However, you can also implement your own solver by inheriting from the :class:`idl.sim.solvers.solver.BaseSolver` class.

.. autoclass:: idl.sim.sim.SIM
   :members:
   :special-members: __call__

Example usage:

.. code-block:: python

   import torch
   from idl.sim import SIM
   from idl.sim.solvers import CVXSolver

   explicit_model = ...
   dataloader = ...

   # Define the SIM model
   sim = SIM(activation_fn=torch.nn.functional.relu, device="cuda", dtype=torch.float32)

   # Define the solver
   solver = CVXSolver()

   # Train SIM
   sim.train(solver=solver, model=explict_model, dataloader=dataloader)

.. toctree::
   :maxdepth: 2
   :caption: Components:

   solvers/solver
   solvers/admm
   solvers/cvx
   solvers/ls
   solvers/gd
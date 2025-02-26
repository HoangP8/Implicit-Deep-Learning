SIM (State-driven Implicit Models)
==================================

State-driven Implicit Modeling (SIM) is an advanced training methodology for implicit models, introduced in `"State-driven Implicit Models" <https://arxiv.org/abs/2209.09389>`_. SIM distills implicit models from pre-trained explicit networks by matching internal state representations.

Theoretical Foundation
----------------------

A standard Implicit Model is defined by:

.. math::
   \begin{aligned}
      X &= \phi(A X + B U) \quad &\text{(Equilibrium equation)}, \\
      \hat{Y} &= C X + D U \quad &\text{(Prediction equation)},
   \end{aligned}

**SIM Training Objective:**

Given:
   - Input matrix :math:`U \in \mathbb{R}^{p \times m}`
   - Synthetic pre-activation states :math:`Z \in \mathbb{R}^{n \times m}` from explicit network
   - Synthetic post-activation states :math:`X \in \mathbb{R}^{n \times m}` from explicit network
   - Synthetic outputs :math:`\hat{Y} \in \mathbb{R}^{q \times m}` from explicit network

SIM solves the following convex optimization problem:

.. math::
   \begin{aligned}
      & \min_{A,B,C,D} \quad f(A,B,C,D)\\
      & \text{subject to:} \\
      & \quad Z = AX + BU, \\
      & \quad \hat{Y} = CX + DU, \\
      & \quad \|A\|_\infty \leq \kappa,
   \end{aligned}

where :math:`f` is an objective function that typically includes regularization terms to promote sparsity or other desirable properties.

Implementation Components
-------------------------

The SIM training process consists of two main phases:

1. **State Extraction**: The :class:`idl.sim.sim.SIM` class contains the method to extract internal state vectors from explicit networks and formulates the optimization problem.

2. **Convex Optimization**: Various solvers are already provided in the next sections to solve the resulting optimization problem efficiently. Moreover, custom solvers can be applied by inheriting from the :class:`idl.sim.solvers.solver.BaseSolver` class.


API Reference
-------------

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
   :maxdepth: 5
   :caption: Subsections:

   solvers/cvx
   solvers/admm
   solvers/ls
   solvers/gd
   solvers/solver
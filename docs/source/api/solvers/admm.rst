ADMM Solvers
------------

Alternating Direction Method of Multipliers (ADMM) is a powerful optimization algorithm for solving constrained convex problems, particularly useful for SIM training.

Theoretical Foundation
^^^^^^^^^^^^^^^^^^^^^^

**Dual Problem Formulation**

Consider the constrained convex optimization problem:

.. math::

   \begin{aligned}
       &\text{minimize} \quad f(x) \\
       &\text{subject to} \quad Ax = b
   \end{aligned}

The dual problem is formulated using:
   - Lagrangian: :math:`L(x, y) = f(x) + y^T (Ax - b)`
   - Dual function: :math:`g(y) = \inf_x L(x, y)`
   - Dual problem: :math:`\text{maximize} \quad g(y)`
   - Primal recovery: :math:`x^\star = \arg\min_x L(x, y^\star)`

**Dual Ascent Method**

To solve the dual problem, we used the Gradient Descent method:

.. math::

   y_{k+1} = y_k + \alpha_k \nabla g(y_k)

where:
   - :math:`\nabla g(y_k) = Ax^\ast - b`
   - :math:`x^\ast = \arg\min_x L(x, y_k)`

This yields the iterative update rules:

.. math::

   \begin{aligned}
       x_{k+1} &:= \arg\min_x L(x, y_k) \\
       y_{k+1} &:= y_k + \alpha_k (Ax_{k+1} - b)
   \end{aligned}

Convergence is achieved when :math:`Ax_{k+1} - b \rightarrow 0`.

**Method of Multipliers**

Powell's augmented Lagrangian introduces a quadratic penalty term with hyperparameter :math:`\rho > 0`:

.. math::

   L_{\rho}(x, y) = f(x) + y^T (Ax - b) + (\rho / 2) \|Ax - b\|_2^2

This leads to modified update rules:

.. math::

   \begin{aligned}
       x_{k+1} &:= \arg\min_x L_{\rho}(x, y_k) \\
       y_{k+1} &:= y_k + \rho (Ax_{k+1} - b)
   \end{aligned}

**ADMM Algorithm**

ADMM extends the Method of Multipliers to problems of the form:

.. math::

   \begin{aligned}
       &\text{minimize} \quad f(x) + g(z) \\
       &\text{subject to} \quad Ax + Bz = c
   \end{aligned}

The augmented Lagrangian becomes:

.. math::

   L_{\rho}(x, z, y) = f(x) + g(z) + y^T (Ax + Bz - c) + (\rho / 2) \|Ax + Bz - c\|_2^2

ADMM applies Gauss-Seidel decomposition to solve for variables separately:

.. math::

   \begin{aligned}
       x_{k+1} &:= \arg\min_x L_{\rho}(x, z_k, y_k)\\
       z_{k+1} &:= \arg\min_z L_{\rho}(x_{k+1}, z, y_k) \\
       y_{k+1} &:= y_k + \rho (Ax_{k+1} + Bz_{k+1} - c)
   \end{aligned}

Convergence is achieved when :math:`Ax_k + Bz_k - c \to 0`.

Application to SIM Training
^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the context of SIM training, ADMM efficiently solves the optimization problem:

.. math::

   \min_{a,b} \frac{1}{2} \|X^T a + U^T b - z\|_2^2 + \lambda \|a, b\|_1

This formulation enables:
   - Row-wise independent optimization
   - Efficient handling of L1 regularization
   - Scalable implementation across multiple GPUs

API Reference
^^^^^^^^^^^^^

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

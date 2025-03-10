ADMM Solvers
------------

Alternating Direction Method of Multipliers (ADMM) is a powerful optimization algorithm for solving constrained convex problems. 
This section presents the derivation of ADMM update rules for SIM training. The algorithm is implemented in :func:`idl.sim.solvers.admm_consensus.ADMMSolver` and :func:`idl.sim.solvers.admm_consensus_multigpu.ADMMMultiGPUSolver`.

Introduction to ADMM
^^^^^^^^^^^^^^^^^^^^

**Dual Problem**

Consider the constrained convex optimization problem:

.. math::

    \text{minimize} \quad f(x)
    
    \text{subject to} \quad Ax = b

The dual approach involves:

- Lagrangian: :math:`L(x, y) = f(x) + y^T (Ax - b)` 
- Dual function: :math:`g(y) = \inf_x L(x, y)` 
- Dual problem: :math:`\text{maximize} \ g(y)` 
- Primal recovery: :math:`x^\star = \arg\min_x L(x, y^\star)`

**Dual Ascent**

To solve the dual problem, we apply the Gradient Ascent method:

.. math::

    y_{k+1} = y_k + \alpha_k \nabla g(y_k)

The gradient of the dual function is:

.. math::

    \nabla g(y_k) = Ax^\ast - b
    
    x^\ast = \arg\min_x L(x, y_k)

Therefore, the update rules to solve the problem are:

.. math::

    x_{k+1} &:= \arg\min_x L(x, y_k) \\
    y_{k+1} &:= y_k + \alpha_k (Ax_{k+1} - b)

We continue updating until :math:`Ax_{k+1} - b \rightarrow 0`.

**Method of Multipliers**

Powell (1969) introduced the **augmented Lagrangian**, with hyperparameter :math:`\rho > 0`:

.. math::

    L_{\rho}(x, y) = f(x) + y^T (Ax - b) + (\rho / 2) \|Ax - b\|_2^2

With this augmented Lagrangian, the update rules become:

.. math::

    x_{k+1} &:= \arg\min_x L_{\rho}(x, y_k) \\
    y_{k+1} &:= y_k + \rho (Ax_{k+1} - b)

We continue updating until :math:`Ax_{k+1} - b \rightarrow 0`.

**Alternating Direction Method of Multipliers (ADMM)**

ADMM addresses problems of the form:

.. math::

    \text{minimize} \quad f(x) + g(z)
    
    \text{subject to} \quad Ax + Bz = c

The augmented Lagrangian, with :math:`\rho > 0`, is:

.. math::

    L_{\rho}(x, z, y) = f(x) + g(z) + y^T (Ax + Bz - c) + (\rho / 2) \|Ax + Bz - c\|_2^2

Instead of solving for :math:`x` and :math:`z` jointly, ADMM applies the Gauss-Seidel method to solve them separately:

.. math::

    x_{k+1} &:= \arg\min_x L_{\rho}(x, z_k, y_k)\\
    z_{k+1} &:= \arg\min_z L_{\rho}(x_{k+1}, z, y_k) \\
    y_{k+1} &:= y_k + \rho (Ax_{k+1} + Bz_{k+1} - c)

We continue updating until :math:`Ax_k + Bz_k - c \to 0`.


State-driven Implicit Modeling (SIM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Implicit Deep Learning Model** 

An implicit model is defined as:

.. math::

    x = \phi(Ax + Bu) \quad \text{[equilibrium equation]}
    
    \hat{y}(u) = Cx + Du \quad \text{[prediction equation]}

**SIM Training Problem Formulation** 

Given an explicit neural network, we can transform it into an implicit model by solving the following optimization problem:

.. math::

    \min_{M} \quad f(M)
    
    \text{s.t.} \quad Z = AX + BU,
    
    \quad \quad \hat{Y} = CX + DU,
    
    \quad \quad \|A\|_\infty \leq \kappa.

Here, :math:`M` is the stacked weight matrix, :math:`M = \begin{pmatrix} A & B \\ C & D \end{pmatrix}`. The matrices :math:`A, B, C, D` are the modified weights of the explicit model. :math:`Z` represents the pre-activation matrix, :math:`U` is the input matrix, and :math:`X` is the post-activation matrix (:math:`X = \sigma(Z)`).

To achieve sparsity, the objective function can be set to :math:`f(M) = \|M\|_\infty = \left\| \begin{pmatrix} A & B \\ C & D \end{pmatrix} \right\|_\infty`.

The problem can be solved row by row, as each row's computation is independent of others, enabling parallel implementation. Moreover, since the problem minimizes the max function in the :math:`l_\infty` norm, it is equivalent to minimizing each element.

We solve for :math:`A, B` first, and then for :math:`C, D` later (the update rules are identical):

.. math::

    \min_{a,b} \quad & f\left(\begin{pmatrix} a \\ b \end{pmatrix}\right) = \left\|\begin{pmatrix} a \\ b \end{pmatrix}\right\|_1 \\
    
    \text{s.t.} \quad & z = \begin{pmatrix} X^T & U^T \end{pmatrix} \begin{pmatrix} a \\ b \end{pmatrix}, \\
    
    & \|a\|_1 \leq \kappa.


Solving SIM with ADMM
^^^^^^^^^^^^^^^^^^^^^

The row-form problem is almost identical to this problem, where we relax the inequality constraint. Note that :math:`\|a\|_1 \leq \left \| \begin{pmatrix} a \\ b \end{pmatrix} \right\|_1 = \|a\|_1 + \|b\|_1 \leq \kappa.`

.. math::

    \min_{a,b} & \quad \frac{1}{2\lambda_1} \left  \| \begin{pmatrix} a \\ b \end{pmatrix} \right\|_1 + \frac{1}{2} \left\| z - \begin{pmatrix} X^T U^T \end{pmatrix} \begin{pmatrix} a \\ b \end{pmatrix} \right\|_2^2 
    
    \text{s.t.}&\quad \left\| \begin{pmatrix} a \\ b \end{pmatrix} \right\|_1 \leq k.

This is equivalent to:

.. math::

    \min_{a,b} & \quad \frac{1}{2} \left\| \begin{pmatrix} X^T U^T \end{pmatrix} \begin{pmatrix} a \\ b \end{pmatrix} - z \right\|_2^2 + \lambda \left  \| \begin{pmatrix} a \\ b \end{pmatrix} \right\|_1
    
    \text{s.t.}&\quad \left\| \begin{pmatrix} a \\ b \end{pmatrix} \right\|_1 \leq k.

**Global consensus form:**

With :math:`\beta = \begin{pmatrix} a \\ b \end{pmatrix}` and :math:`\Phi = \begin{pmatrix} X^T U^T \end{pmatrix}`, we express the problem in global consensus form:

.. math::

    \min & \quad \frac{1}{2}\|\Phi \beta_1 - z\|_2^2 + \lambda \|\beta_2\|_1 + I_C(\beta_3) \\
    \text{s.t.} & \quad \beta_1 = \beta_2 = \beta_3 \\

**ADMM update rules:**

.. math::

    \beta_1^{k+1} &= \arg\min_{\beta_1} \left( \frac{1}{2}\|\Phi \beta_1 - z\|_2^2 + \frac{\rho}{2} \|\beta_1 - \bar{\beta}^k + u_1^k\|_2^2 \right)
    
    \beta_2^{k+1} &= \arg\min_{\beta_2} \left( \lambda \|\beta_2\|_1 + \frac{\rho}{2} \|\beta_2 - \bar{\beta}^k + u_2^k\|_2^2 \right)
    
    \beta_3^{k+1} &= \arg\min_{\beta_3} \left( I_C(\beta_3) + \frac{\rho}{2} \|\beta_3 - \bar{\beta}^k + u_3^k\|_2^2 \right)
    
    \bar{\beta}^{k+1} &= \frac{1}{3} \sum_{i=1}^{3} \beta_i^k
    
    u_i^{k+1} &= u_i^k + \beta_i^{k+1} - \bar{\beta}^{k+1} \quad (i = 1,2,3)

The closed-form update rules are derived as:

.. math::

    \beta_1^{k+1} &= \left( \Phi^T \Phi + \rho I \right)^{-1} \left( \Phi^T z + \rho \left( \bar{\beta}^k - u_1^k \right) \right)
    
    \beta_2^{k+1} &= \mathcal{S} \left( \bar{\beta}^{k} - u_2^k, \frac{\lambda}{\rho} \right), \quad \text{where } \mathcal{S}(z, a) = z - \max \left( \min(z, a), -a \right)
    
    \beta_3^{k+1} &= \text{Proj}_C (\bar{\beta}^{k} - u_3^k)

The projection operation can be performed efficiently by projecting the first n rows of :math:`(\bar{\beta}^{k} - u_3^k)` onto the norm ball with radius :math:`\kappa` and copying the remaining elements of :math:`(\bar{\beta}^{k} - u_3^k)` to :math:`\beta_3^{k+1}`.

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

ADMM Solvers
------------

.. autoclass:: idl.sim.solvers.admm_consensus_multigpu.ADMMMultiGPUSolver
   :members: solve

.. autoclass:: idl.sim.solvers.admm_consensus.ADMMSolver
   :members: solve

Introduction
============

Dual problem
------------

Consider the constrained convex optimization problem:

.. math::

   \begin{aligned}
       &\text{minimize} \quad f(x) \\
       &\text{subject to} \quad Ax = b
   \end{aligned}

- Lagrangian: :math:`L(x, y) = f(x) + y^T (Ax - b)`
- Dual function: :math:`g(y) = \inf_x L(x, y)`
- Dual problem: :math:`\text{maximize} \quad g(y)`
- Recover :math:`x^\star = \arg\min_x L(x, y^\star)`

Dual ascent
-----------

To solve the dual problem, we used the Gradient Descent method:

.. math::

   y_{k+1} = y_k + \alpha_k \nabla g(y_k)

It is easy to see that:

.. math::

   \nabla g(y_k) = Ax^\ast - b \\
   x^\ast = \arg\min_x L(x, y_k)

Therefore, we have update rules to solve the problem:

.. math::

   \begin{aligned}
       x_{k+1} &:= \arg\min_x L(x, y_k) \\
       y_{k+1} &:= y_k + \alpha_k (Ax_{k+1} - b)
   \end{aligned}

We update until :math:`Ax_{k+1} - b \rightarrow 0`.

Method of Multipliers
---------------------

Powell :cite:`Powell1969AMF` introduced **augmented Lagrangian**, with hyperparameter :math:`\rho > 0`:

.. math::

   L_{\rho}(x, y) = f(x) + y^T (Ax - b) + (\rho / 2) \|Ax - b\|_2^2

With the new **augmented Lagrangian**, we have new update rules:

.. math::

   \begin{aligned}
       x_{k+1} &:= \arg\min_x L_{\rho}(x, y_k) \\
       y_{k+1} &:= y_k + \rho (Ax_{k+1} - b)
   \end{aligned}

We update until :math:`Ax_{k+1} - b \rightarrow 0`.

Alternating Direction Method of Multipliers (ADMM)
---------------------------------------------------

ADMM problem's form:

.. math::

   \begin{aligned}
       &\text{minimize} \quad f(x) + g(z) \\
       &\text{subject to} \quad Ax + Bz = c
   \end{aligned}

The **augmented Lagrangian**, with :math:`\rho > 0`:

.. math::

   L_{\rho}(x, z, y) = f(x) + g(z) + y^T (Ax + Bz - c) + (\rho / 2) \|Ax + Bz - c\|_2^2

Instead of solving :math:`x, z` jointly, we solve them separately (Gauss-Seidel method). Therefore, update rules are:

.. math::

   \begin{aligned}
       x_{k+1} &:= \arg\min_x L_{\rho}(x, z_k, y_k)\\
       z_{k+1} &:= \arg\min_z L_{\rho}(x_{k+1}, z, y_k) \\
       y_{k+1} &:= y_k + \rho (Ax_{k+1} + Bz_{k+1} - c)
   \end{aligned}

We update until :math:`Ax_k + Bz_k - c \to 0`.

Implicit Deep Learning
======================

Instead of using recursive formulas as in traditional neural networks, we consider implicit prediction rules:

.. math::

   \begin{aligned}
       x &= \phi(Ax + Bu) \quad \text{[equilibrium equation]} \\
       \hat{y}(u) &= Cx + Du \quad \text{[prediction equation]}
   \end{aligned}

State-driven Implicit Modeling
------------------------------

### Training

To achieve better compression with minimal performance loss, we solve:

.. math::

   \begin{aligned}
       & \min_{M} \quad f(M)\\
       & \text{s.t.} \quad
       Z = AX + BU, \\
       & \quad \quad \hat{Y} = CX + DU, \\
       & \quad \quad \|A\|_\infty \leq \kappa.
   \end{aligned}

where :math:`M = \begin{pmatrix} A & B \\ C & D \end{pmatrix}` :cite:`tsai2022statedrivenimplicitmodelingsparsity`.

### Inference

After training, given :math:`U`, solve :math:`X = \sigma(AX + BU)`, then compute :math:`\hat{Y} = CX + DU`.

SIM Training Setup
------------------

We minimize the max function using the :math:`l_\infty` norm:

.. math::

   \min_{a,b} \|a\|_1 + \|b\|_1 + \lambda_1 \|z - X^T a - U^T b\|_2^2

where each row is solved independently.

ADMM in SIM
===========

Using the ADMM method, we solve:

.. math::

   \min_{a,b} \frac{1}{2} \|X^T a + U^T b - z\|_2^2 + \lambda \|a, b\|_1

using consensus form and iterative updates :cite:`boyd2011distributed`.

References
==========

.. bibliography:: references.bib

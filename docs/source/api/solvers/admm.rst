ADMM Solvers
------------

Alternating Direction Method of Multipliers (ADMM) is a powerful optimization algorithm for solving constrained convex problems, particularly useful for SIM training.

Introduction
------------

**Dual problem**

Consider the constrained convex optimization problem:

.. math::

    \text{minimize} \quad f(x)
    
    \text{subject to} \quad Ax = b

* Lagrangian: :math:`L(x, y) = f(x) + y^T (Ax - b)`
* dual function: :math:`g(y) = \inf_x L(x, y)`
* dual problem: :math:`\text{maximize} \quad g(y)`
* recover :math:`x^\star = \arg\min_x L(x, y^\star)`

**Dual ascent**

To solve the dual problem, we used the Gradient Descent method:

.. math::

    y_{k+1} = y_k + \alpha_k \nabla g(y_k)

It is easy to see that:

.. math::

    \nabla g(y_k) = Ax^\ast - b
    
    x^\ast = \arg\min_x L(x, y_k)

Therefore, we have update rules to solve problem in section 1.1 as below:

.. math::

    x_{k+1} &:= \arg\min_x L(x, y_k) \\
    y_{k+1} &:= y_k + \alpha_k (Ax_{k+1} - b)

We update until :math:`Ax_{k+1} - b \rightarrow 0`

**Method of Multipliers**

Powell (1969) introduced **augmented lagranian**, with hyperparameter :math:`\rho > 0`.

.. math::

    L_{\rho}(x, y) = f(x) + y^T (Ax - b) + (\rho / 2) \|Ax - b\|_2^2

With new **augmented lagranian**, we have new update rules as:

.. math::

    x_{k+1} &:= \arg\min_x L_{\rho}(x, y_k) \\
    y_{k+1} &:= y_k + \rho (Ax_{k+1} - b)

Here, We update until :math:`Ax_{k+1} - b \rightarrow 0`

**Alternating direction of method of multipliers (ADMM)**

ADMM problem's form, note that :math:`f, g` are convex

.. math::

    \text{minimize} \quad f(x) + g(z)
    
    \text{subject to} \quad Ax + Bz = c

The **augmented Lagranian**, with :math:`\rho > 0`:

.. math::

    L_{\rho}(x, z, y) = f(x) + g(z) + y^T (Ax + Bz - c) + (\rho / 2) \|Ax + Bz - c\|_2^2

Instead of solving :math:`x, z` jointly, we can solve it separately (Gauss-Seidel method). Therefore, we have an update rules:

.. math::

    x_{k+1} &:= \arg\min_x L_{\rho}(x, z_k, y_k)\\
    z_{k+1} &:= \arg\min_z L_{\rho}(x_{k+1}, z, y_k) \\
    y_{k+1} &:= y_k + \rho (Ax_{k+1} + Bz_{k+1} - c)

Here, we update until :math:`Ax_k + Bz_k - c \to 0`

===================
Implicit Deep Learning
===================

Instead of using recursive formulas as in the traditional neural network, we consider the new class of deep learning model that based on implicit prediction rules:
We have two mains equations:

.. math::

    x = \phi(Ax + Bu) \quad \text{[equilibrium equation]}
    
    \hat{y}(u) = Cx + Du \quad \text{[prediction equation]}

Note that we will always have a way to transfer the traditional neural network into the implicit neural network, the details can be found in section 3 of the paper (Ghaoui et al., 2020).

State-driven Implicit Modeling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Training**

The paper (Tsai et al., 2022) only contains two basic neural network architectures, the Fully-connected Neural Network, and Convolutional Neural Network.

In order to achieve the better compression (Have less non-zero parameters) with the hope that the performance of the model will not change. We have the objective problem as below.

.. math::

    \min_{M} \quad f(M)
    
    \text{s.t.} \quad Z = AX + BU,
    
    \quad \quad \hat{Y} = CX + DU,
    
    \quad \quad \|A\|_\infty \leq \kappa.

In which the :math:`M` is the stack weight matrix, :math:`M = \begin{pmatrix} A & B \\ C & D \end{pmatrix}`. And :math:`A, B, C, D` are the modified weight matrix of the explicit model (the fully-connected one). :math:`Z` is the pre-activation matrix, :math:`U` is the input matrix, :math:`X` is the post-activation matrix (:math:`X = \sigma(Z)`).

The paper (Tsai et al., 2022) also presented another way to train the weight matrix :math:`M` by introducing the *Relax states and output matching* problem:

.. math::

    \min_M \quad & f(M) + \lambda_1 \| Z - (AX + BU) \|_F^2 + \lambda_2 \| \hat{Y} - (CX + DU) \|_F^2
    
    \text{s.t.} \ \|A\|_\infty \leq \kappa

**Inference**

After achieving the :math:`A, B, C, D` from training phase, we calculate :math:`X` given :math:`U` by solving the fixed point equation :math:`X = \sigma(AX + BU)` (If :math:`A` is wellposed to :math:`\sigma`, or inequality satisfies, this equation has unique solution), and then calculate the :math:`\hat{Y} = CX + DU`.

SIM training setup
^^^^^^^^^^^^^^^^^

To achieve the sparsity, we can let the :math:`f(M) = \|M\|_\infty = \| \begin{pmatrix} A & B \\ C & D \end{pmatrix}\|_\infty` rather than use the :math:`l_1` norm. Since the :math:`l_1` norm and the :math:`l_\infty` are almost identical in achieving the sparsity. Moreover, the :math:`l_\infty` helps us solve the objective problem in vector form more easily.

Here, we can solve the problem row by row, since each row's computation is independent of others (We can benefit from parallel implementation). Moreover, the problem minimizes the max function in the :math:`l_\infty` norm, which is identical to minimizing each element.

We will solve A, B first, and C, and D later (Note that the update rules are identical):

.. math::

    \min_{a,b} \quad & f(\begin{pmatrix} a \\ b \end{pmatrix}) =  \|\begin{pmatrix} a \\ b \end{pmatrix}\|_1
    
    \text{s.t.} \quad & z = \begin{pmatrix} X^T & U^T \end{pmatrix} \begin{pmatrix} a \\ b \end{pmatrix},
    
    & \|a\|_1 \leq \kappa.

Similarly, we have the row-form of the *Relax states and output matching* problem as below:

.. math::

    \min_{a,b} \|a\|_1 + \|b\|_1 + \lambda_1 \left\| z - X^T a - U^T b \right\|_2^2
    
    \text{subject to} \quad \| a \|_{1} \leq \kappa

With :math:`\begin{pmatrix} a \\ b \end{pmatrix}` is the arbitrary column of :math:`\begin{pmatrix} A^T \\ B^T \end{pmatrix}`.

Apply similarly, we have the row-form for problem of :math:`C, D`.

==========
ADMM in SIM
==========

The row-form problem is almost identical to this problem, we relax the inequality constraint. Note that we have :math:`\|a\|_1 \leq \left \| \begin{pmatrix} a \\ b \end{pmatrix} \right\|_1 = \|a\|_1 + \|b\|_1 \leq \kappa.`

.. math::

    \min_{a,b} & \quad \frac{1}{2\lambda_1} \left  \| \begin{pmatrix} a \\ b \end{pmatrix} \right\|_1 + \frac{1}{2} \left\| z - \begin{pmatrix} X^T U^T \end{pmatrix} \begin{pmatrix} a \\ b \end{pmatrix} \right\|_2^2
    
    \text{subject to}&\quad \left\| \begin{pmatrix} a \\ b \end{pmatrix} \right\|_1 \leq k.

Make a slight change, we have:

**Original problem:**

.. math::

    \min_{a,b} & \quad \frac{1}{2} \left\| \begin{pmatrix} X^T U^T \end{pmatrix} \begin{pmatrix} a \\ b \end{pmatrix} - z \right\|_2^2 + \lambda \left  \| \begin{pmatrix} a \\ b \end{pmatrix} \right\|_1
    
    \text{s.t.}&\quad \left\| \begin{pmatrix} a \\ b \end{pmatrix} \right\|_1 \leq k.

**Global consensus form:**

With :math:`\beta = \begin{pmatrix} a \\ b \end{pmatrix}` and :math:`\Phi = \begin{pmatrix} X^T U^T \end{pmatrix}` we have the following global consensus form:

.. math::

    \min & \quad \frac{1}{2}\|\Phi \beta_1 - z\|_2^2 + \lambda \|\beta_2\|_1 + I_C(\beta_3)
    
    \text{s.t.} & \quad \beta_1 = \beta_2 = \beta_3
    
    & \quad C = \{\beta \mid \|(I_n,0)\beta\|_1 \leq \kappa\}

**ADMM update rules:**

.. math::

    \beta_1^{k+1} &= argmin_{\beta_1} \left( \frac{1}{2}\|\Phi \beta_1 - z\|_2^2 + \frac{\rho}{2} \|\beta_1 - \bar{\beta}^k + u_1^k\|_2^2 \right)
    
    \beta_2^{k+1} &= argmin_{\beta_2} \left( \lambda \|\beta_2\|_1 + \frac{\rho}{2} \|\beta_2 - \bar{\beta}^k + u_2^k\|_2^2 \right)
    
    \beta_3^{k+1} &= argmin_{\beta_3} \left( I_C(\beta_3) + \frac{\rho}{2} \|\beta_3 - \bar{\beta}^k + u_3^k\|_2^2 \right)
    
    \bar{\beta}^{k+1} &= \frac{1}{3} \sum_{i=1}^{3} \beta_i^k
    
    u_i^{k+1} &= u_i^k + \beta_i^{k+1} - \bar{\beta}^{k+1} \quad (i = 1,2,3)

The closed form update rules are derived as below:

.. math::

    \beta_1^{k+1} &= \left( \Phi^T \Phi + \rho I \right)^{-1} \left( \Phi^T z + \rho \left( \bar{\beta}^k - u_1^k \right) \right)
    
    \beta_2^{k+1} &= \mathcal{S} \left( \bar{\beta}^{k} - u_2^k, \frac{\lambda}{\rho} \right), \quad \text{note that } \mathcal{S}(z, a) = z - \max \left( \min(z, a), -a \right)
    
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

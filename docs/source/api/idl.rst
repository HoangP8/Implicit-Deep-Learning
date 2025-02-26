Implicit Models
===============

Implicit Deep Learning introduces a novel class of models based on fixed-point prediction rules, as formalized in `"Implicit Deep Learning" <https://arxiv.org/abs/1908.06315>`_.

Theoretical Foundation
----------------------

Given the following dimensions:
   - :math:`p`: input features dimension
   - :math:`q`: output features dimension
   - :math:`n`: hidden state dimension
   - :math:`m`: batch size

An implicit model processes an input matrix :math:`U \in \mathbb{R}^{p \times m}` to produce an output matrix :math:`\hat{Y} \in \mathbb{R}^{q \times m}` by solving:

.. math::
   \begin{aligned}
      X &= \phi(A X + B U) \quad &\text{(Equilibrium equation)}, \\
      \hat{Y} &= C X + D U \quad &\text{(Prediction equation)},
   \end{aligned}

where:
   - :math:`A \in \mathbb{R}^{n \times n}, B \in \mathbb{R}^{n \times p}, C \in \mathbb{R}^{q \times n}, D \in \mathbb{R}^{q \times p}` are learnable parameters
   - :math:`X \in \mathbb{R}^{n \times m}` is the equilibrium hidden state
   - :math:`\phi: \mathbb{R}^{n \times m} \to \mathbb{R}^{n \times m}` is a non-linear activation function (default: ReLU)

Low-Rank Parameterization
-------------------------

For memory efficiency and improved generalization, the weight matrix :math:`A` can be parameterized as a low-rank product:

.. math::
   A = L R^T

where :math:`L, R \in \mathbb{R}^{n \times r}` with rank :math:`r \ll n`.

Well-Posedness Condition
------------------------

To guarantee the existence and uniqueness of solutions to the equilibrium equation, the model must satisfy:

.. math::
   \|A\|_\infty < \kappa, \quad \text{where } 0 \leq \kappa < 1

This constraint ensures the fixed-point iteration converges to a unique solution.

API Reference
-------------

.. autoclass:: idl.implicit_base_model.ImplicitModel
   :members:
   :undoc-members:

Example usage:

.. code-block:: python

   import torch
   from idl import ImplicitModel
   
   x = torch.randn(5, 64)  # (batch_size=5, input_dim=64)
   
   model = ImplicitModel(input_dim=64,  
                        output_dim=10, 
                        hidden_dim=128)
   
   output = model(x)  # (batch_size=5, output_dim=10)

A special case of Implicit Model for Recurrent Neural Networks is provided in the following section.

.. toctree::
   :maxdepth: 2
   :caption: Subsections:

   rnn
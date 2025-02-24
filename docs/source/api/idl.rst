Implicit Models
===============

Base Implicit model from `"Implicit Deep Learning" <https://arxiv.org/abs/1908.06315>`_, 
a new class of deep learning models based on implicit prediction rules.

Specifically, given the following dimensions:
   - :math:`p`: number of input features,
   - :math:`q`: number of output features,
   - :math:`n`: number of hidden features,
   - :math:`m`: batch size (number of samples per batch),

the implicit model operates on an input matrix :math:`U \in \mathbb{R}^{p \times m}` and predicts an output matrix :math:`\hat{Y} \in \mathbb{R}^{q \times m}`, by solving the following equations:

.. math::
   \begin{aligned}
      X &= \phi(A X + B U) \quad &\text{(Equilibrium equation)}, \\
      \hat{Y} &= C X + D U \quad &\text{(Prediction equation)},
   \end{aligned}

where:
   - :math:`A \in \mathbb{R}^{n \times n}, B \in \mathbb{R}^{n \times p}, C \in \mathbb{R}^{q \times n}, D \in \mathbb{R}^{q \times p}` are learnable parameters,
   - :math:`X \in \mathbb{R}^{n \times m}` is the hidden state of the implicit model,
   - :math:`\hat{Y} \in \mathbb{R}^{q \times m}` is the predicted output,
   - :math:`\phi: \mathbb{R}^{n \times m} \to \mathbb{R}^{n \times m}` is an activation function (default is ReLU).
   
For **low-rank approximation** of the implicit model, the matrix :math:`A` is calculated as:

.. math::
   A = L R^T

where :math:`L, R \in \mathbb{R}^{n \times r}` with :math:`r \ll n`.
   
To ensure the fixed-point equaion has an unique solution, the **wellposedness** of implicit model must be satisfied, which means

.. math::
   0 \leq \left\Vert A \right\Vert_\infty < \kappa, \quad \text{where } 0 \leq \kappa < 1


.. autoclass:: idl.implicit_base_model.ImplicitModel
   :members:
   :undoc-members:

A special case of Implicit Model for Recurrent Neural Networks is provided in the following section.

.. toctree::
   :maxdepth: 2
   :caption: Components:

   rnn
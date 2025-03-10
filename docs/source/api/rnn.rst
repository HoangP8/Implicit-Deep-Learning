Implicit RNN
============

Implicit Recurrent Neural Network extend the implicit modeling framework to sequential data processing. Unlike traditional RNNs that update hidden states using explicit linear transformations,
`ImplicitRNN` uses an implicit layer to define recurrence in a standard RNN framework.

Theoretical Foundation
----------------------

Given the following dimensions:
   - :math:`p`: input features dimension
   - :math:`q`: output features dimension
   - :math:`n`: hidden state dimension
   - :math:`T`: sequence length

An Implicit RNN processes a sequence of inputs :math:`\{u_t\}_{t=1}^T` where :math:`u_t \in \mathbb{R}^p` to produce a sequence of outputs :math:`\{\hat{y}_t\}_{t=1}^T` where :math:`\hat{y}_t \in \mathbb{R}^q` by solving:

.. math::
   \begin{aligned}
      X_t &= \phi(A X_t + B [U_t, H_{t-1}]) \quad &\text{(Equilibrium equation)}, \\
      H_t &= C X_t + D U_t \quad &\text{(Hidden state update)}.
   \end{aligned}

The final hidden state :math:`H_T` is projected to the output:

.. math::
   \hat{Y} = \text{Linear}(H_T),

where:
   - :math:`A \in \mathbb{R}^{n \times n}, B \in \mathbb{R}^{n \times (p+n)}, C \in \mathbb{R}^{n \times n}, D \in \mathbb{R}^{n \times p}` are learnable parameters,
   - :math:`U_t \in \mathbb{R}^{m \times p}` is the input at timestep :math:`t`,
   - :math:`X_t \in \mathbb{R}^{m \times n}` is the implicit hidden state solved via a fixed-point equation,
   - :math:`H_t \in \mathbb{R}^{m \times n}` is the RNN hidden state at timestep :math:`t`,
   - :math:`\phi: \mathbb{R}^{n \times m} \to \mathbb{R}^{n \times m}` is an activation function (default is ReLU),
   - :math:` \text{Linear}` is a linear transformation that turns :math:`H_T` into :math:`\hat{Y}`.

API Reference
-------------

.. autoclass:: idl.implicit_rnn_model.ImplicitRNN
   :members:
   :undoc-members:

Example usage:

.. code-block:: python

   import torch
   from idl import ImplicitRNN
   
   x = torch.randn(100, 60, 1)  # (batch_size=100, seq_len=60, input_dim=1)
   
   model = ImplicitRNN(input_dim=1,  
                       output_dim=1, 
                       hidden_dim=128, 
                       implicit_hidden_dim=64)
   output = model(x)  # (batch_size=100, output_dim=1)
   
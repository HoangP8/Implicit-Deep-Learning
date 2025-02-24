Implicit RNN Model
===================

Implicit Recurrent Neural Network (`ImplicitRNN`) for sequence modeling. Unlike traditional RNNs that update hidden states using explicit linear transformations,
`ImplicitRNN` uses an implicit layer to define recurrence in a standard RNN framework.

Specifically, given the following dimensions:
   - :math:`p`: number of input features,
   - :math:`q`: number of output features,
   - :math:`n`: number of hidden features,
   - :math:`m`: batch size (number of samples per batch),
   - :math:`T`: sequence length,

the implicit hidden state :math:`X_t \in \mathbb{R}^{m \times n}` and the RNN hidden state :math:`H_t \in \mathbb{R}^{m \times n}`
are computed by solving the following equations:

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
   - :math:`\phi: \mathbb{R}^{n \times m} \to \mathbb{R}^{n \times m}` is an activation function (default is ReLU).

.. autoclass:: idl.implicit_rnn_model.ImplicitRNN
   :members:
   :undoc-members:
from .implicit_base_model import ImplicitModel
import torch
from torch import nn, Tensor
from typing import Any, Tuple


class ImplicitRNNCell(nn.Module):
    def __init__(
        self,
        input_dim: int,
        implicit_hidden_dim: int,
        hidden_dim: int,
        **kwargs: Any
    ) -> None:
        """
        An RNN cell that utilizes an implicit model for recurrent operations.
        
        Mimics a vanilla RNN, but where the recurrent operation is performed by an implicit model instead of a linear layer.
        Follows the "batch first" convention, where the input x has shape (batch_size, seq_len, input_dim).
        The hidden state doubles as the output, but it is not recommended to use it directly as the model's prediction
        for low-dimensionality problems, since it is responsible for passing information between timesteps. Instead, it is
        recommended to use a larger hidden state with a linear layer that scales it down to the desired output dimension.
        
        Args:
            input_dim (int): Dimensionality of the input features.
            implicit_hidden_dim (int): Number of hidden features in the underlying implicit model.
            hidden_dim (int): Dimensionality of the recurrent hidden state.
            **kwargs: Additional keyword arguments for the underlying ImplicitModel.
        """
        
        super().__init__()
        self.hidden_dim: int = hidden_dim
        self.layer: ImplicitModel = ImplicitModel(
            hidden_dim=implicit_hidden_dim, 
            input_dim=input_dim + hidden_dim, 
            output_dim=hidden_dim, 
            **kwargs
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of the ImplicitRNNCell.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            Tuple[Tensor, Tensor]:
                - outputs (Tensor): Hidden states for all timesteps, shape (batch_size, seq_len, hidden_dim).
                - h (Tensor): Final hidden state, shape (batch_size, hidden_dim).
        """

        outputs = torch.empty(*x.shape[:-1], self.hidden_dim, device=x.device)
        
        h = torch.zeros(x.shape[0], self.hidden_dim, device=x.device)
        for t in range(x.shape[1]):
            h = self.layer(torch.cat((x[:, t, :], h), dim=-1))
            outputs[:, t, :] = h
            
        return outputs, h
    

class ImplicitRNN(nn.Module):
    r""" 
    Implicit Recurrent Neural Networks.
    
    Args:
        input_dim (int): Number of input features (:math:`p`).
        output_dim (int): Number of output features (:math:`q`).
        implicit_hidden_dim (int): Hidden dimension in the implicit layer (:math:`n`).
        hidden_dim (int): Size of the recurrent hidden state (:math:`n`).
        **kwargs (Any): Additional keyword arguments for `ImplicitModel`.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        implicit_hidden_dim: int,
        hidden_dim: int,
        **kwargs: Any
    ) -> None:
        
        super().__init__()
        self.recurrent: ImplicitRNNCell = ImplicitRNNCell(
            input_dim=input_dim,
            implicit_hidden_dim=implicit_hidden_dim,
            hidden_dim=hidden_dim,
            **kwargs
        )
        self.linear: nn.Linear = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of `ImplicitRNN`.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            Tensor: Output tensor of shape (batch_size, output_dim).
        """

        _, h = self.recurrent(x)
        return self.linear(h)

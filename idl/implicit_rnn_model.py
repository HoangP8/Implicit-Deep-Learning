from .implicit_base_model import ImplicitModel
import torch
from torch import nn


class ImplicitRNNCell(nn.Module):
    def __init__(self, input_dim, implicit_hidden_dim, hidden_dim, **kwargs):
        """
        Create a new ImplicitRNNCell:
            Mimics a vanilla RNN, but where the recurrent operation is performed by an implicit model instead of a linear layer.
            Follows the "batch first" convention, where the input x has shape (batch_size, seq_len, input_size).
            The hidden state doubles as the output, but it is not recommended to use it directly as the model's prediction
            for low-dimensionality problems, since it is responsible for passing information between timesteps. Instead, it is
            recommended to use a larger hidden state with a linear layer that scales it down to the desired output dimension.
        
        Args:
            input_size: the input dimension into the RNN cell.
            n: the number of hidden features in the underlying implicit model (equivalent to n in the ImplicitModel class).
            hidden_size: the recurrent hidden dimension of the ImplicitRNNCell. Equivalent to the hidden size in a GRU, for example.
            **kwargs: other keyword arguments passed to the undelying ImplicitModel.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.layer = ImplicitModel(hidden_dim=implicit_hidden_dim, 
                                    input_dim=input_dim+hidden_dim, 
                                    output_dim=hidden_dim, 
                                    **kwargs)

    def forward(self, x):
        outputs = torch.empty(*x.shape[:-1], self.hidden_dim, device=x.device)
        
        h = torch.zeros(x.shape[0], self.hidden_dim, device=x.device)
        for t in range(x.shape[1]):
            h = self.layer(torch.cat((x[:, t, :], h), dim=-1))
            outputs[:, t, :] = h
            
        return outputs, h
    

class ImplicitRNN(nn.Module):
    def __init__(self, input_dim, implicit_hidden_dim, hidden_dim, output_dim, **kwargs):
        super().__init__()
        self.recurrent = ImplicitRNNCell(input_dim, implicit_hidden_dim, hidden_dim, **kwargs)
        self.linear = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        _, h = self.recurrent(x)
        return self.linear(h)
    
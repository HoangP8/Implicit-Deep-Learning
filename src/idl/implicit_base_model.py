import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Type
from .implicit_function import ImplicitFunctionInf, ImplicitFunction, transpose, project_onto_Linf_ball


class ImplicitModel(nn.Module):
    r""" 
    **Note**: In conventional deep learning, the batch size typically comes first for inputs :math:`U`, hidden states :math:`X`, and outputs :math:`Y`. 
    We follow this convention, but the model internally transposes these matrices to solve the fixed-point equation.
    Users can input their data in the usual format, and the output will be returned in the standard format. Below is an example:

    >>> import torch
    >>> from idl import ImplicitModel
    >>> 
    >>> x = torch.randn(5, 64)  # (batch_size=5, input_dim=64)
    >>> 
    >>> model = ImplicitModel(input_dim=64,  
    >>>                       output_dim=10, 
    >>>                       hidden_dim=128)
    >>> 
    >>> output = model(x)  # (batch_size=5, output_dim=10)

    Args:
        input_dim (int): Number of input features (:math:`p`).
        output_dim (int): Number of output features (:math:`q`).
        hidden_dim (int): Number of hidden features (:math:`n`).
        is_low_rank (bool, optional): Whether to use low-rank approximation (default: False).
        rank (int, optional): Rank for low-rank approximation (:math:`r`), required if `is_low_rank` is True.
        f (Type[ImplicitFunction], optional): The implicit function to use (default: ImplicitFunctionInf for well-posedness).
        kappa (float, optional): Radius of the L-infinity norm ball (:math:`\kappa`) for well-posedness. (default: 0.99).
        no_D (bool, optional): Whether to exclude matrix D (default: False).
        bias (bool, optional): Whether to include a bias term (default: False).
        mitr (int, optional): Max iterations for the forward pass. (default: 300).
        grad_mitr (int, optional): Max iterations for gradient computation. (default: 300).
        tol (float, optional): Convergence tolerance for the forward pass. (default: 3e-6).
        grad_tol (float, optional): Convergence tolerance for gradients. (default: 3e-6).
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        f: Type[ImplicitFunction] = ImplicitFunctionInf,
        no_D: bool = False,
        bias: bool = False,
        mitr: int = 300,
        grad_mitr: int = 300,
        tol: float = 3e-6,
        grad_tol: float = 3e-6,
        kappa: float = 0.99,
        is_low_rank: bool = False,
        rank: Optional[int] = None
    ) -> None:
        super().__init__()

        if is_low_rank and rank is None:
            raise ValueError("Parameter 'k' is required when 'is_low_rank' is True.")

        if bias:
            input_dim += 1

        self.hidden_dim: int = hidden_dim
        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        self.is_low_rank: bool = is_low_rank
        self.no_D: bool = no_D
        self.bias: bool = bias
        
        if self.is_low_rank:
            if rank is None:
                raise ValueError("Rank must be specified when using low-rank approximation.")
            self.L: nn.Parameter = nn.Parameter(torch.randn(hidden_dim, rank) / hidden_dim)
            self.R: nn.Parameter = nn.Parameter(torch.randn(hidden_dim, rank) / hidden_dim)
        else:
            self.A: nn.Parameter = nn.Parameter(torch.randn(hidden_dim, hidden_dim) / hidden_dim)

        self.B: nn.Parameter = nn.Parameter(torch.randn(hidden_dim, input_dim) / hidden_dim)
        self.C: nn.Parameter = nn.Parameter(torch.randn(output_dim, hidden_dim) / hidden_dim)
        if not self.no_D:
            self.D: nn.Parameter = nn.Parameter(torch.randn(output_dim, input_dim) / hidden_dim)
        else:
            self.D: torch.Tensor = torch.zeros((output_dim, input_dim), requires_grad=False)

        self.f: ImplicitFunction = f()
        self.f.set_parameters(mitr=mitr, grad_mitr=grad_mitr, tol=tol, grad_tol=grad_tol, kappa=kappa)


    def forward(
        self, 
        U: torch.Tensor, 
        X0: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of `ImplicitModel`.

        Args:
            U (torch.Tensor): Input tensor of shape (batch_size, input_dim).
            X0 (torch.Tensor, optional): Initial hidden state tensor of shape (batch_size, hidden_dim).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, output_dim).
        """

        if (len(U.size()) == 3):
            U = U.flatten(1, -1)
        U = transpose(U)
        if self.bias:
            U = F.pad(U, (0, 0, 0, 1), value=1)
        assert U.shape[0] == self.input_dim, f'Given input size {U.shape[0]} does not match expected input size {self.p}.'

        m = U.shape[1]
        X_shape = torch.Size([self.hidden_dim, m])

        if X0 is not None:
            X0 = transpose(X0)
            assert X0.shape == X_shape
        else:
            X0 = torch.zeros(X_shape, dtype=U.dtype, device=U.device)

        if self.is_low_rank:
            L_projected = project_onto_Linf_ball(self.L, self.f.kappa)
            RT_projected = project_onto_Linf_ball(transpose(self.R), self.f.kappa)
            X = self.f.apply(L_projected @ RT_projected, self.B, X0, U)
            
        else:
            X = self.f.apply(self.A, self.B, X0, U)
        return transpose(self.C @ X + self.D @ U)
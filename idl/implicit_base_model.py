import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
from .implicit_function import ImplicitFunctionInf, ImplicitFunction, transpose, project_onto_Linf_ball


class ImplicitModel(nn.Module):
    """
    Creates an Implicit Model with:
        A: hidden_dim * hidden_dim
        B: hidden_dim * input_dim
        C: output_dim * hidden_dim
        D: output_dim * input_dim (if `no_D` is False)
        X: hidden_dim * batch_size
        U: input_dim * batch_size

    Note that for X and U, the batch size comes first when inputting into the model.
    These sizes reflect that the model internally transposes them so that their sizes line up with ABCD.

    Args:
        hidden_dim (int): Number of hidden features.
        input_dim (int): Number of input features.
        output_dim (int): Number of output features.
        f (ImplicitFunction, optional): The implicit function to use (default: ImplicitFunctionInf for well-posedness).
        no_D (bool, optional): Whether to exclude matrix D (default: False).
        bias (bool, optional): Whether to include a bias term (default: False).
        mitr (int, optional): Max iterations for the forward pass. (default: 300).
        grad_mitr (int, optional): Max iterations for gradient computation. (default: 300).
        tol (float, optional): Convergence tolerance for the forward pass. (default: 3e-6).
        grad_tol (float, optional): Convergence tolerance for gradients. (default: 3e-6).
        v (float, optional): Radius of the L-infinity norm ball for projection. (default: 0.95).
        is_low_rank (bool, optional): Whether to use low-rank approximation (default: False).
        rank (int, optional): Rank for low-rank approximation (required if `is_low_rank` is True).
    """

    def __init__(self, hidden_dim: int, input_dim: int, output_dim: int,
                 f: Optional[ImplicitFunction] = ImplicitFunctionInf,
                 no_D: Optional[bool] = False,
                 bias: Optional[bool] = False,
                 mitr: Optional[int] = 300,
                 grad_mitr: Optional[int] = 300,
                 tol: Optional[float] = 3e-6,
                 grad_tol: Optional[float] = 3e-6,
                 v: Optional[float] = 0.95,
                 is_low_rank: Optional[bool] = False,
                 rank: Optional[int] = None):
        super(ImplicitModel, self).__init__()

        if is_low_rank and rank is None:
            raise ValueError("Parameter 'k' is required when 'is_low_rank' is True.")

        if bias:
            input_dim += 1

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_low_rank = is_low_rank
        
        if self.is_low_rank:
            self.L = nn.Parameter(torch.randn(hidden_dim, rank)/hidden_dim)
            self.R = nn.Parameter(torch.randn(hidden_dim, rank)/hidden_dim)     
        else:
            self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim)/hidden_dim)   
            
        self.B = nn.Parameter(torch.randn(hidden_dim, input_dim)/hidden_dim)
        self.C = nn.Parameter(torch.randn(output_dim, hidden_dim)/hidden_dim)
        self.D = nn.Parameter(torch.randn(output_dim, input_dim)/hidden_dim) if not no_D else torch.zeros((output_dim, input_dim), requires_grad=False)
        
        self.f = f  # The class of the implicit function (e.g., ImplicitFunctionInf)
        self.f.set_parameters(mitr=mitr, grad_mitr=grad_mitr, tol=tol, grad_tol=grad_tol, v=v)
        self.bias = bias


    def forward(self, U: torch.Tensor, X0: Optional[torch.Tensor] = None):
        """
        Performs a forward pass of the implicit model.

        Args:
            U (torch.Tensor): Input tensor of shape (batch_size, input_dim) or (batch_size, ..., input_dim).
            X0 (torch.Tensor, optional): Initial hidden state tensor of shape (hidden_dim, batch_size).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, output_dim).

        Process:
            1. Transposes and processes the input `U` for compatibility with model dimensions.
            2. Handles optional bias padding if `bias` is True.
            3. Initializes or validates the initial hidden state `X0`.
            4. Computes the hidden state `X` using the implicit function:
                - Uses low-rank approximation if `is_low_rank` is True.
                - Otherwise, applies the full weight matrix.
            5. Computes the output as a combination of `C @ X` and `D @ U`, transposing back to batch-first format.
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
            L_projected = project_onto_Linf_ball(self.L, self.f.v)
            RT_projected = project_onto_Linf_ball(transpose(self.R), self.f.v)
            X = self.f.apply(L_projected @ RT_projected, self.B, X0, U)
            
        else:
            X = self.f.apply(self.A, self.B, X0, U)
        return transpose(self.C @ X + self.D @ U)
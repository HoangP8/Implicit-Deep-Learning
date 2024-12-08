import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
from .implicit_function import ImplicitFunctionInf, ImplicitFunction, transpose, project_onto_Linf_ball


class ImplicitModel(nn.Module):
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
        """
        Create a new Implicit Model:
            A: n*n  B: n*p  C: q*n  D: q*p
            X: n*m  U: p*m, m = batch size
            Note that for X and U, the batch size comes first when inputting into the model.
            These sizes reflect that the model internally transposes them so that their sizes line up with ABCD.
        
        Args:
            n: the number of hidden features.
            p: the number of input features.
            q: the number of output classes.
            f: the implicit function to use.
            no_D: whether or not to use the D matrix (i.e., whether the prediction equation should explicitly depend on the input U).
            bias: whether or not to use a bias.
        """
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
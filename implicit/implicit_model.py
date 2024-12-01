import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
from .implicit_function import ImplicitFunctionInf, ImplicitFunctionTriu

def transpose(X):
    """
    Convenient function to transpose a matrix.
    """
    assert len(X.size()) == 2, "data must be 2D"
    return X.T


class ImplicitModel(nn.Module):
    def __init__(self, hidden_size: int, input_size: int, output_size: int,
                 f: str = 'ImplicitFunctionInf',
                 no_D: Optional[bool] = False,
                 bias: Optional[bool] = False,
                 mitr: Optional[int] = 300,
                 grad_mitr: Optional[int] = 300,
                 tol: Optional[float] = 3e-6,
                 grad_tol: Optional[float] = 3e-6,
                 activation: str = 'relu'):
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

        if bias:
            input_size += 1

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        
        self.A = nn.Parameter(torch.randn(hidden_size, hidden_size)/hidden_size)
        self.B = nn.Parameter(torch.randn(hidden_size, input_size)/hidden_size)
        self.C = nn.Parameter(torch.randn(output_size, hidden_size)/hidden_size)
        self.D = nn.Parameter(torch.randn(output_size, input_size)/hidden_size) if not no_D else torch.zeros((output_size, input_size), requires_grad=False)
        
        self.f = self._get_function_class(f)  # The class of the implicit function (e.g., ImplicitFunctionInf)
        self.f.set_parameters(mitr=mitr, grad_mitr=grad_mitr, tol=tol, grad_tol=grad_tol, activation=activation)
        self.bias = bias

    def _get_function_class(self, class_name: str):
        """Load the class based on string name."""
        class_map = {
            'ImplicitFunctionInf': ImplicitFunctionInf,
            'ImplicitFunctionTriu': ImplicitFunctionTriu
        }
        
        return class_map.get(class_name, ImplicitFunctionInf)

    def forward(self, U: torch.Tensor, X0: Optional[torch.Tensor] = None):
        U = transpose(U)
        if self.bias:
            U = F.pad(U, (0, 0, 0, 1), value=1)
        assert U.shape[0] == self.input_size, f'Given input size {U.shape[0]} does not match expected input size {self.p}.'

        m = U.shape[1]
        X_shape = torch.Size([self.hidden_size, m])

        if X0 is not None:
            X0 = transpose(X0)
            assert X0.shape == X_shape
        else:
            X0 = torch.zeros(X_shape, dtype=U.dtype, device=U.device)

        X = self.f.apply(self.A, self.B, X0, U)

        return transpose(self.C @ X + self.D @ U)
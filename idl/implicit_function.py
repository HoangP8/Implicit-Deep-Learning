import torch
from torch.autograd import Function
import numpy as np
import warnings
import torch

def transpose(X):
    """
    Transpose a 2D matrix.
    """

    assert len(X.size()) == 2, "data must be 2D"
    return X.T


def project_onto_Linf_ball(A, v):
    """
    Project a matrix onto the L-infty norm ball of radius v.

    Args:
        A (torch.Tensor): Input matrix.
        v (float): The L-infty norm ball radius.

    Returns:
        torch.Tensor: The scaled matrix.
    """

    norm_inf_A = torch.linalg.matrix_norm(A, ord=float('inf')) 
    if norm_inf_A > v:
        A = (v / norm_inf_A) * A
    return A


class ImplicitFunctionWarning(RuntimeWarning):
    pass

class ImplicitFunction(Function):
    """
    Base class for solving implicit functions of the fixed-point equation `AX + BU = X`.
    """
    
    @classmethod
    def set_parameters(cls, mitr=None, grad_mitr=None, tol=None, grad_tol=None, v=None):
        """
        Set parameters for Picard iteration and convergence.

        Args:
            mitr (int): Max iterations for the forward pass.
            grad_mitr (int): Max iterations for gradient computation.
            tol (float): Convergence tolerance for the forward pass.
            grad_tol (float): Convergence tolerance for gradients.
            v (float): Radius of the L-infinity norm ball for projection.
        """

        if mitr is not None:
            cls.mitr = mitr
        if grad_mitr is not None:
            cls.grad_mitr = grad_mitr
        if tol is not None:
            cls.tol = tol
        if grad_tol is not None:
            cls.grad_tol = grad_tol
        if v is not None:
            cls.v = v                    
    
    @classmethod
    def forward(cls, ctx, A, B, X0, U):
        """
        Perform the forward pass using Picard iterations to solve `AX + BU = X`.

        Args:
            A (torch.Tensor)
            B (torch.Tensor)
            X0 (torch.Tensor)
            U (torch.Tensor)

        Returns:
            torch.Tensor: The stable solution for X
        """

        with torch.no_grad():
            X, err, status = cls.inn_pred(A, B @ U, X0, cls.mitr, cls.tol)
        ctx.save_for_backward(A, B, X, U)
        if status != "converged":
            warnings.warn(f"Picard iterations did not converge: err={err.item():.4e}, status={status}", ImplicitFunctionWarning)
        return X

    @classmethod
    def backward(cls, ctx, *grad_outputs):
        """
        Compute gradients using implicit backward propagation for the fixed-point equation.

        Args:
            grad_outputs (tuple): Gradients of the output.

        Returns:
            tuple: 
                - Gradient of A.
                - Gradient of B.
                - Zero gradient for X (since X is an output).
                - Gradient of U.
        """

        A, B, X, U = ctx.saved_tensors

        grad_output = grad_outputs[0]
        assert grad_output.size() == X.size()

        DPhi = cls.dphi(A @ X + B @ U)
        V, err, status = cls.inn_pred_grad(A.T, DPhi * grad_output, DPhi, cls.grad_mitr, cls.grad_tol)
        if status != "converged":
            warnings.warn(f"Gradient iterations did not converge: err={err.item():.4e}, status={status}", ImplicitFunctionWarning)
        
        grad_A = V @ X.T
        grad_B = V @ U.T
        grad_U = B.T @ V

        return grad_A, grad_B, torch.zeros_like(X), grad_U

    @staticmethod
    def phi(X):
        """
        ReLU activation.
        """

        return torch.clamp(X, min=0)

    @staticmethod
    def dphi(X):
        """
        Derivative of ReLU.
        """

        grad = X.new_zeros(X.shape)
        grad[X > 0] = 1
        return grad

    @classmethod
    def inn_pred(cls, A, Z, X, mitr, tol):
        """
        Solve `AX + Z = X` using Picard iterations.

        Args:
            A (torch.Tensor)
            Z (torch.Tensor)
            X (torch.Tensor)
            mitr (int): Max iterations.
            tol (float): Convergence tolerance.

        Returns:
            tuple: Solution, error, status ('converged' or 'max itrs reached').
        """

        err = 0
        status = 'max itrs reached'
        for _ in range(mitr):
            X_new = cls.phi(A @ X + Z)
            err = torch.norm(X_new - X, np.inf)
            if err < tol:
                status = 'converged'
                break
            X = X_new
        return X, err, status

    @staticmethod
    def inn_pred_grad(AT, Z, DPhi, mitr, tol):
        """
        Compute gradient using backward Picard iterations.

        Args:
            AT (torch.Tensor): Transposed matrix A.
            Z (torch.Tensor)
            mitr (int): Max iterations.
            tol (float): Convergence tolerance.
            DPhi (torch.Tensor): Derivative of activation.

        Returns:
            tuple: Gradient, error, status ('converged' or 'max itrs reached').
        """
    
        X = torch.zeros_like(Z)
        err = 0
        status = 'max itrs reached'
        for _ in range(mitr):
            X_new = DPhi * (AT @ X) + Z
            err = torch.norm(X_new - X, np.inf)
            if err < tol:
                status = 'converged'
                break
            X = X_new
        return X, err, status


class ImplicitFunctionTriu(ImplicitFunction):
    """
    Implicit function that constrains A to be upper triangular, ensuring a feed-forward architecture.
    
    Methods:
        forward: Applies upper triangular constraint to A and runs the forward pass.
        backward: Keeps the gradient of A upper triangular.
    """

    @classmethod
    def forward(cls, ctx, A, B, X0, U):
        A = A.triu_(1)
        return super(ImplicitFunctionTriu, cls).forward(ctx, A, B, X0, U)

    @classmethod
    def backward(cls, ctx, *grad_outputs):
        grad_A, grad_B, grad_X, grad_U = super(ImplicitFunctionTriu, cls).backward(ctx, *grad_outputs)
        return grad_A.triu(1), grad_B, grad_X, grad_U


class ImplicitFunctionInf(ImplicitFunction):
    """
    Implicit function that projects A onto the infinity norm ball, enabling closed-loop feedback.
    
    Methods:
        forward: Projects A onto the infinity norm ball and runs the forward pass.
    """

    @classmethod
    def forward(cls, ctx, A, B, X0, U):
        A = project_onto_Linf_ball(A, cls.v)
        return super(ImplicitFunctionInf, cls).forward(ctx, A, B, X0, U)
import torch
from torch.autograd import Function
import numpy as np
import warnings
import torch

class ImplicitFunctionWarning(RuntimeWarning):
    pass

class ImplicitFunction(Function):
    mitr = grad_mitr = 300
    tol = grad_tol = 3e-6
    activation = 'relu'
    
    @classmethod
    def set_parameters(cls, mitr=None, grad_mitr=None, tol=None, grad_tol=None, activation=None):
        if mitr is not None:
            cls.mitr = mitr
        if grad_mitr is not None:
            cls.grad_mitr = grad_mitr
        if tol is not None:
            cls.tol = tol
        if grad_tol is not None:
            cls.grad_tol = grad_tol
        if activation is not None:
            cls.activation = activation

    @classmethod
    def forward(cls, ctx, A, B, X0, U):
        with torch.no_grad():
            X, err, status = cls.inn_pred(A, B @ U, X0, cls.mitr, cls.tol)
        ctx.save_for_backward(A, B, X, U)
        if status != "converged":
            warnings.warn(f"Picard iterations did not converge: err={err.item():.4e}, status={status}", ImplicitFunctionWarning)
        return X

    @classmethod
    def backward(cls, ctx, *grad_outputs):
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

    @classmethod
    def phi(cls, X):
        """ Activation function phi (either ReLU or SiLU based on class-level activation setting). """
        
        if ImplicitFunction.activation == 'relu':
            return torch.clamp(X, min=0)
        elif ImplicitFunction.activation == 'silu':
            return X * torch.sigmoid(X)
        else:
            raise ValueError(f"Unknown activation function: {ImplicitFunction.activation}")

    @classmethod
    def dphi(cls, X):
        """ Derivative of the activation function dphi (for ReLU or SiLU). """
        
        if ImplicitFunction.activation == 'relu':
            grad = X.new_zeros(X.shape)
            grad[X > 0] = 1
            return grad
        elif ImplicitFunction.activation == 'silu':
            grad = X.clone().detach()
            sigmoid = torch.sigmoid(grad)
            return sigmoid * (1 + grad * (1 - sigmoid))
        else:
            raise ValueError(f"Unknown activation function: {ImplicitFunction.activation}")

    @classmethod
    def inn_pred(cls, A, Z, X, mitr, tol):
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
    Constrains the A matrix to be upper triangular. Only allows for the implicit model to learn feed-forward architectures.
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
    Implicit function which projects A onto the infinity norm ball. Allows for the model to learn closed-loop feedback.
    """

    @classmethod
    def forward(cls, ctx, A, B, X0, U):

        # project A on |A|_inf=v
        v = 0.95

        norm_inf_A = torch.linalg.matrix_norm(A, ord=float('inf')) 
        if (norm_inf_A > v):
            A = (v * A) / norm_inf_A
        else:
            pass
        
        return super(ImplicitFunctionInf, cls).forward(ctx, A, B, X0, U)
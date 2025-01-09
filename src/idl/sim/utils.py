"""
Utility functions for SIM
"""

import logging

import gc
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def csr_to_sparse_tensor(csr, device):
    """
    Convert scipy csr_matrix to torch.sparse_coo_tensor
    """
    coordinate_tensor = csr.tocoo()
    indices = np.vstack((coordinate_tensor.row, coordinate_tensor.col))
    values = coordinate_tensor.data
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    return torch.sparse_coo_tensor(i, v, csr.shape).to(device)


@torch.no_grad()
def fixpoint_iteration(A, B, U, activation_fn, device, max_iter=50000, atol=1e-6, patience=1000):
    """
    Compute X = phi(A @ X + B @ U) until convergence and return X, with support for sparse matrices.
    """

    # Convert scipy sparse matrices to PyTorch sparse tensors
    if isinstance(A, np.ndarray):
        A=torch.tensor(A,device=device)
        B=torch.tensor(B,device=device)
    else:
        A = csr_to_sparse_tensor(A, device)
        B = csr_to_sparse_tensor(B, device)

    # Compute the infinite norm of matrix A
    inf_norm_A = torch.linalg.norm(A.to_dense(), np.inf)
    logger.info(f"Infinite norm of A: {inf_norm_A}")

    # Convert numpy array U to PyTorch tensor
    if not torch.is_tensor(U):
        U = torch.from_numpy(U).float().to(device, dtype=B.dtype)
    else:
        U = U.float().to(device, dtype=B.dtype)

    # Initialize X with zeros on GPU
    n, m = A.shape[-1], U.shape[-1]
    X = torch.zeros((n, m), device=device, dtype=A.dtype)
    next_X = activation_fn(torch.sparse.mm(A, X) + torch.sparse.mm(B, U))

    min_diff = np.inf
    iteration = 0
    while True:
        X = next_X.clone()
        next_X = activation_fn(torch.sparse.mm(A, next_X) + torch.sparse.mm(B, U))

        # Compute the infinite norm of the difference
        difference = torch.linalg.norm(next_X - X, np.inf).item()
        if difference < atol:
            logger.info(f"Fixpoint iteration converged at Iteration {iteration}")
            break

        if iteration % 1000 == 0:
            logger.info(f"Iteration {iteration}: Difference = {difference}")

        # Early stopping condition
        if difference >= min_diff:
            if patience <= 0:
                logger.info(f"Early stopping at Iteration {iteration}: Difference = {difference}")
                break
            patience -= 1
        else:
            min_diff = difference
            patience = 1000  # reset patience if there is improvement

        if iteration >= max_iter:
            logger.info("Fixpoint iteration did not converge")
            break

        iteration += 1

    return next_X

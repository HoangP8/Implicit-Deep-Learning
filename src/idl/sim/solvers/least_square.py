import logging
import numpy as np
from typing import Any, Dict, Tuple

from .solver import BaseSolver
from ..utils import fixpoint_iteration

logger = logging.getLogger(__name__)

class LeastSquareSolver(BaseSolver):
    r"""
    Solve using numpy.linalg.lstsq. 
    Note: This solver is fast but it cannot handle the wellposeness condition (norm(A) <= kappa).

    Args:
        regen_states (bool, optional): Whether to regenerate state data to solve exact C,D after solving A,B. Defaults to False.
        tol (float, optional): Zero out weights that are less than tol. Defaults to 1e-6.
    """
    def __init__(
        self, 
        regen_states : bool = False,
        tol : float = 1e-6,
    ):
        self.regen_states = regen_states
        self.tol = tol

    def solve(
        self, 
        X : np.ndarray, 
        U : np.ndarray, 
        Z : np.ndarray, 
        Y : np.ndarray, 
        model_config : Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve with numpy.linalg.lstsq to get an implicit model.

        Args:
            X (np.ndarray): Post-activation array.
            U (np.ndarray): Input array.
            Z (np.ndarray): Pre-activation array.
            Y (np.ndarray): Output array.
            model_config (Dict[str, Any]): Model configuration. 
                - activation_fn (Callable): Activation function used by the implicit model.
                - device (str): Implicit model's device.
                - atol (float): Equilibrium function's tolerance.
                - kappa (float): Wellposedness condition parameter.
                
        Returns:
            A, B, C, D (np.ndarray): Implicit model's parameters.
        """
        n, m, p, q = X.shape[0], X.shape[1], U.shape[0], Y.shape[0]

        logger.info("===== Start solving A and B =====")
        AB = self.solve_matrix(np.hstack([X.T, U.T]), Z.T)
        A = AB[:, :n]
        B = AB[:, n:]

        if self.regen_states:
            X = fixpoint_iteration(A, B, U, model_config['activation_fn'], model_config['device'], atol=model_config['atol']).cpu()

        logger.info("===== Start solving C and D =====")
        CD = self.solve_matrix(np.hstack([X.T, U.T]), Y.T)
        C = CD[:, :n]
        D = CD[:, n:]

        return A, B, C, D

    def solve_matrix(self, X : np.ndarray, Y : np.ndarray) -> np.ndarray:
        """
        Wrapper to solve a least square problem.

        Args:
            X (np.ndarray): Input array.
            Y (np.ndarray): Output array.
        Returns:
            W (np.ndarray): Weight array.
        """
        
        W, c, r, _ = np.linalg.lstsq(X, Y, rcond=None)

        loss = np.mean(np.square(X @ W - Y))
        
        logger.info(f"Total Lasso loss: {loss}")
        logger.info(f"Data rank: {r}")

        W[np.abs(W) <= self.tol] = 0
        
        return W.T
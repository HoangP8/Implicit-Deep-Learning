import gc
import logging
import numpy as np

from ..utils import fixpoint_iteration

logger = logging.getLogger(__name__)

class LeastSquareSolver:
    def __init__(
            self, 
            regen_states : bool = False,
            tol : float = 1e-6,
    ):
        """
        Solve using numpy.linalg.lstsq. 
        Note: This solver is fast but it cannot handle the wellposeness condition.

        Args:
            regen_states (bool, optional): Whether to regenerate states data. Defaults to False.
            tol (float, optional): Zero out weights that are less than tol. Defaults to 1e-6.
        """
        self.regen_states = regen_states
        self.tol = tol

    def solve(self, X, U, Z, Y, config):
        n, m, p, q = X.shape[0], X.shape[1], U.shape[0], Y.shape[0]

        logger.info("===== Start solving A and B =====")
        AB = self.solve_matrix(np.hstack([X.T, U.T]), Z.T, is_y=False, n=n)
        A = AB[:, :n]
        B = AB[:, n:]

        if self.regen_states:
            X = fixpoint_iteration(A, B, U, config['activation_fn'], config['device'], atol=config['atol']).cpu()

        logger.info("===== Start solving C and D =====")
        CD = self.solve_matrix(np.hstack([X.T, U.T]), Y.T, is_y=True, n=n)
        C = CD[:, :n]
        D = CD[:, n:]

        return A, B, C, D

    def solve_matrix(self, X, Y):
        
        W, c, r, _ = np.linalg.lstsq(X, Y, rcond=None)

        loss = np.mean(np.square(X @ W - Y))
        
        logger.info(f"Total Lasso loss: {loss}")
        logger.info(f"Data rank: {r}")

        W[np.abs(W) <= self.tol] = 0
        
        return W.T
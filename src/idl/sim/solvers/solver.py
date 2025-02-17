from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np

class Solver(ABC):
    r"""
    Base class for all solvers. All solver implementations must inherit from this class
    and implement the solve method.
    """
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def solve(
        self, 
        X: np.ndarray, 
        U: np.ndarray, 
        Z: np.ndarray, 
        Y: np.ndarray, 
        model_config: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Solve for the implicit model parameters.

        Args:
            X (np.ndarray): Post-activation array of shape (n_samples, hidden_dim).
            U (np.ndarray): Input array of shape (n_samples, input_dim).
            Z (np.ndarray): Pre-activation array of shape (n_samples, hidden_dim).
            Y (np.ndarray): Output array of shape (n_samples, output_dim).
            model_config (Dict[str, Any]): Model configuration containing:
                - activation_fn (Callable): Activation function used by the implicit model
                - device (str): Device to run computations on ('cpu' or 'cuda')
                - atol (float): Absolute tolerance for convergence
                - kappa (float): Wellposedness condition parameter

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - A (np.ndarray): Hidden-to-hidden weight matrix of shape (hidden_dim, hidden_dim)
                - B (np.ndarray): Input-to-hidden weight matrix of shape (hidden_dim, input_dim)
                - C (np.ndarray): Hidden-to-output weight matrix of shape (output_dim, hidden_dim)
                - D (np.ndarray): Input-to-output weight matrix of shape (output_dim, input_dim)
        """
        raise NotImplementedError
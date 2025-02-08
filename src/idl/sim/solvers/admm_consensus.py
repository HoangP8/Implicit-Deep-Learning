import logging
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, Optional, Tuple

from ..utils import fixpoint_iteration

logger = logging.getLogger(__name__)


class ADMMSolver:
    def __init__(
        self, 
        num_epoch_ab : int = 1000,
        num_epoch_cd : int = 100,
        lambda_y : float = 1e-6,
        lambda_z : float = 1e-6,
        rho_ab : float = 10.0,
        rho_cd : float = 10.0,
        batch_feature_size : int = 100,
        regen_states : bool = False,
        tol : float = 1e-6,
    ):
        """
        ADMM Consensus Solver on a single GPU.

        Args:
            num_epoch_ab (int, optional): Number of epochs for solving A and B. Defaults to 1000.
            num_epoch_cd (int, optional): Number of epochs for solving C and D. Defaults to 100.
            lambda_y (float, optional): Lasso regularization parameter for Y. Defaults to 1.0.
            lambda_z (float, optional): Lasso regularization parameter for Z. Defaults to 1.0.
            rho_ab (float, optional): ADMM's rho parameter for A and B. Defaults to 10.0.
            rho_cd (float, optional): ADMM's rho parameter for C and D. Defaults to 10.0.
            batch_feature_size (int, optional): Number of columns to solve in each solving iteration. This is used to control the memory usage.
                    The solver is performed (total_rows // batch_feature_size + 1) times. Defaults to 100.
            regen_states (bool, optional): Whether to regenerate states. Defaults to False.
            tol (float, optional): Tolerance for zeroing out weights. Defaults to 1e-6.
        """
        self.num_epoch_ab = num_epoch_ab
        self.num_epoch_cd = num_epoch_cd
        self.lambda_y = lambda_y
        self.lambda_z = lambda_z
        self.rho_ab = rho_ab
        self.rho_cd = rho_cd
        self.batch_feature_size = batch_feature_size
        self.regen_states = regen_states
        self.tol = tol

    def solve(
        self, 
        X : np.ndarray, 
        U : np.ndarray, 
        Z : np.ndarray, 
        Y : np.ndarray,
        model_config : Dict[str, Any],
        plot_loss : Optional[bool] = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve State-driven Implicit Model using the ADMM consensus algorithm.

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
            plot_loss (Optional[bool], optional): Whether to plot loss curve. Useful for tuning hyperparameters and debugging. Defaults to False.

        Returns:
            A, B, C, D (np.ndarray): Implicit model's parameters.
        """
        n, m, p, q = X.shape[0], X.shape[1], U.shape[0], Y.shape[0]

        X, U, Z, Y = torch.tensor(X), torch.tensor(U), torch.tensor(Z), torch.tensor(Y)

        logger.info("===== Start solving A and B =====")
        AB = self.parallel_solve_matrix(torch.hstack([X.T, U.T]), Z.T, is_y=False, n=n, kappa=model_config['kappa'], plot_loss=plot_loss, device=model_config['device'])
        A = AB[:, :n]
        B = AB[:, n:]

        if self.regen_states:
            X = fixpoint_iteration(A, B, U, model_config['activation_fn'], model_config['device'], atol=model_config['atol']).cpu()

        logger.info("===== Start solving C and D =====")
        CD = self.parallel_solve_matrix(torch.hstack([X.T, U.T]), Y.T, is_y=True, n=n, kappa=model_config['kappa'], plot_loss=plot_loss, device=model_config['device'])
        C = CD[:, :n]
        D = CD[:, n:]

        return A, B, C, D


    def parallel_solve_matrix(self, X, Y, is_y, n, kappa, plot_loss, device):
        """
        Dividing the data matrix for memory controlled solve.
        """
        total_rows = Y.shape[1]
        batch_rows_length = self.batch_feature_size
        num_batches = total_rows // (batch_rows_length) + 1

        W = None
        loss = 0.0
        for k in range(num_batches):

            logger.info(f"Solving batch feature {k+1}/{num_batches}")

            start_idx = k * batch_rows_length
            end_idx = min((k + 1) * batch_rows_length, total_rows)
            Y_batch = Y[:, start_idx:end_idx]
            
            W_k, loss_k = self.run_solve_opt_problem(X, Y_batch, is_y, n, k, kappa, plot_loss=plot_loss, device=device)
            
            W = np.vstack([W, W_k]) if W is not None else W_k

            loss += loss_k
        
        logger.info(f"Total Lasso loss: {loss}")
        
        return W


    def run_solve_opt_problem(self, X, Y, is_y, n, k, kappa, plot_loss, device):
        """
        ADMM Solve Wrapper.
        """
        if is_y:
            num_epoch = self.num_epoch_cd
            rho = self.rho_cd
            lambda_yz = self.lambda_y
        else:
            num_epoch = self.num_epoch_ab
            rho = self.rho_ab
            lambda_yz = self.lambda_z

        if is_y:
            admm = ADMM_CD(X.shape[1], Y.shape[1], rho, lambda_yz, device=device)
        else:
            admm = ADMM_AB(X.shape[1], Y.shape[1], n, rho, lambda_yz, kappa, device=device)

        losses = []
        with torch.no_grad():
            for i in tqdm(range(num_epoch)):
                admm.step(X, Y)

                # admm.rho = rho * (config.admm.rho_schedule) ** i
            
                losses.append(admm.LassoObjective(X, Y))

        # Plot losses 
        if plot_loss:
            plt.figure()
            plt.plot(losses)
            plt.xlabel("Epoch")
            plt.ylabel("Lasso Objective")
            plt.yscale("log")
            plt.title(f"Training Loss")
            plt.savefig(f"loss_k_{k}_isy_{is_y}.png")

            # Save the loss trace
            np.save(f"loss_trace_k_{k}_isy_{is_y}.npy", losses)

        if is_y:
            result = admm.CD.T.clone().detach().cpu().numpy()
        else:
            result = admm.avg.T.clone().detach().cpu().numpy()

        result[np.abs(result) <= self.tol] = 0

        return result, losses[-1]


class ADMM_AB:
    def __init__(
        self, 
        D : int, 
        Q : int, 
        n : int, 
        rho : float, 
        lambda_yz : float, 
        kappa : float, 
        device : str
    ):
        """
        Solver class to solve A and B (concatenated) using consensus ADMM.

        Args:
            D (int): Number of features.
            Q (int): Number of samples.
            n (int): Number of rows of A.
            rho (float): ADMM's rho parameter.
            lambda_yz (float): Lasso regularization parameter.
            kappa (float): Wellposedness condition parameter.
            device (str): Device.
        """
        self.D = D
        self.Q = Q
        self.n = n
        self.device = device
        
        self.nu_AB = torch.zeros(self.D, self.Q, device=device, requires_grad=False) # ADMM's dual variable for AB
        self.nu_Z = torch.zeros(self.D, self.Q, device=device, requires_grad=False) # ADMM's dual variable for Z
        self.nu_M = torch.zeros(self.D, self.Q, device=device, requires_grad=False) # ADMM's dual variable for M

        self.rho = rho

        self.AB = torch.randn(self.D, self.Q, device=device, requires_grad=False) # running estimate of AB
        self.Z = torch.zeros(self.D, self.Q, device=device, requires_grad=False) # Z
        self.M = torch.zeros(self.D, self.Q, device=device, requires_grad=False) # M
        self.avg = torch.zeros(self.D, self.Q, device=device, requires_grad=False) # Average of AB, Z, M => Final solution of AB

        self.lambda_yz = lambda_yz
        self.kappa = kappa

    @torch.no_grad()
    def step(self, X, y):
        """
        ADMM's update step.
        """
        X = X.to(self.device)
        y = y.to(self.device)

        t1 = X.T.matmul(X) + self.rho * torch.eye(self.D, device=self.device)
        t2 = X.T.matmul(y) + self.rho * (self.avg - self.nu_AB)
        self.AB = torch.linalg.solve(t1, t2)

        self.Z = torch.sign(self.avg - self.nu_Z) * torch.clamp(torch.abs(self.avg - self.nu_Z) - self.lambda_yz / self.rho, min=0)

        self.M[:self.n,:] = self.project_w((self.avg - self.nu_M)[:self.n,:].T).T
        self.M[self.n:,:] = (self.avg - self.nu_M)[self.n:,:]

        self.avg = (self.AB + self.Z + self.M) / 3

        self.nu_AB = self.nu_AB + (self.AB - self.avg)
        self.nu_Z = self.nu_Z + (self.Z - self.avg)
        self.nu_M = self.nu_M + (self.M - self.avg)

    def project_w(self, matrix):
        """
        Project the matrix to the L1 norm ball.
        """
        A_np = matrix.clone().cpu().numpy()
        v = self.kappa
        x = np.abs(A_np).sum(axis=-1)

        for idx in np.where(x > v)[0]:
            a_orig = A_np[idx, :]
            a_sign = np.sign(a_orig)
            a_abs = np.abs(a_orig)
            a = np.sort(a_abs)

            s = np.sum(a) - v
            l = float(len(a))
            for i in range(len(a)):
                if s / l > a[i]:
                    s -= a[i]
                    l -= 1
                else:
                    break
            alpha = s / l if l > 0 else np.max(a_abs)
            a = a_sign * np.maximum(a_abs - alpha, 0)
            # assert np.isclose(np.abs(a).sum(), v)
            A_np[idx, :] = a

        proj = torch.tensor(A_np, dtype=self.avg.dtype, device=self.device)

        return proj

    @torch.no_grad()
    def LassoObjective(self, X, y):
        """
        Evaluate the Lasso objective.
        """
        X = X.to(self.device)
        y = y.to(self.device)
        return (0.5 * torch.norm(X.matmul(self.avg) - y)**2 + self.lambda_yz * torch.sum(torch.abs(self.avg))).item()


class ADMM_CD:
    def __init__(
        self, 
        D : int, 
        Q : int, 
        rho : float, 
        lambda_yz : float, 
        device : str
    ):
        """
        Solver class to solve C and D (concatenated) using ADMM.

        Args:
            D (int): Number of features.
            Q (int): Number of samples.
            rho (float): ADMM's rho parameter.
            lambda_yz (float): Lasso regularization parameter.
            device (str): Device.
        """
        self.D = D
        self.Q = Q
        self.device = device
        
        self.nu = torch.zeros(self.D, self.Q, device=device)
        self.rho = rho

        self.CD = torch.randn(self.D, self.Q, device=device)
        self.Z = torch.zeros(self.D, self.Q, device=device)
        self.lambda_yz = lambda_yz

    @torch.no_grad()
    def step(self, X, y):
        """
        ADMM's update step.
        """
        X = X.to(self.device)
        y = y.to(self.device)

        t1 = X.T.matmul(X) + self.rho * torch.eye(self.D, device=self.device)
        t2 = X.T.matmul(y) + self.rho * self.Z - self.nu
        self.CD = torch.linalg.solve(t1, t2)

        self.Z = self.CD + self.nu / self.rho - (self.lambda_yz / self.rho) * torch.sign(self.Z).to(self.device)
        self.nu = self.nu + self.rho * (self.CD - self.Z)

    @torch.no_grad()
    def LassoObjective(self, X, y):
        """
        Evaluate the Lasso objective.
        """
        X = X.to(self.device)
        y = y.to(self.device)
        return (0.5 * torch.norm(X.matmul(self.CD) - y)**2 + self.lambda_yz * torch.sum(torch.abs(self.CD))).item()

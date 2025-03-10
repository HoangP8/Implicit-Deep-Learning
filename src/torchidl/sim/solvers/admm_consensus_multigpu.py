import gc
import logging
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Any, Dict, List, Optional, Union, Tuple

from .solver import BaseSolver
from ..utils import fixpoint_iteration

logger = logging.getLogger(__name__)


class ADMMMultiGPUSolver(BaseSolver):
    r"""
    ADMM Consensus Solver with distributed computation over multiple GPUs.
    This is highly recommended for large-scale problems and when multiple GPUs are available.

    Args:
        gpu_ids (List[Union[int, torch.device]]): List of GPU IDs or devices.
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
    def __init__(
        self,
        gpu_ids : List[Union[int, torch.device]],
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
        super().__init__()
        self.gpu_ids = gpu_ids
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
        AB = parallel_solve_matrix(torch.hstack([X.T, U.T]), Z.T, is_y=False, n=n, kappa=model_config["kappa"], plot_loss=plot_loss)
        A = AB[:, :n]
        B = AB[:, n:]

        if self.regen_states:
            X = fixpoint_iteration(A, B, U, model_config["activation_fn"], model_config["device"], atol=model_config["atol"]).cpu()

        logger.info("===== Start solving C and D =====")
        CD = parallel_solve_matrix(torch.hstack([X.T, U.T]), Y.T, is_y=True, n=n, kappa=model_config["kappa"], plot_loss=plot_loss)
        C = CD[:, :n]
        D = CD[:, n:]

        return A, B, C, D

    def parallel_solve_matrix(self, X, Y, is_y, n, kappa, plot_loss):
        """
        Dividing the data matrix and distribute batches for parallel solving on multiple GPUs.
        """
        world_size = len(self.gpu_ids)
        total_rows = Y.shape[1]
        batch_rows_length = self.batch_feature_size
        num_batches = total_rows // (batch_rows_length * world_size) + 1

        W = None
        loss = 0.0
        for k in range(num_batches):

            logger.info(f"Solving batch feature {k+1}/{num_batches}")

            start_idx = k * batch_rows_length * world_size
            end_idx = min((k + 1) * batch_rows_length * world_size, total_rows)
            Y_batch = Y[:, start_idx:end_idx]

            # Multiprocessing: spawn multiple processes with torch.multiprocessing
            manager = mp.Manager()
            return_dict = manager.dict()

            mp.set_start_method('spawn', force=True)
            processes = []
            for rank in range(world_size):
                p = mp.Process(target=run_solve_opt_problem, args=(rank, world_size, X, Y_batch, batch_rows_length, is_y, n, kappa, plot_loss, return_dict))
                processes.append(p)
                p.start()
            for p in processes:
                p.join()
            
            # Aggregate results from all processes
            results = [return_dict[i][0] for i in range(world_size)]
            W_k = np.vstack(results)
            W = np.vstack([W, W_k]) if W is not None else W_k

            loss += np.sum([return_dict[i][1] for i in range(world_size)])

            del W_k, results, return_dict
            gc.collect()
        
        logger.info(f"Total Lasso loss: {loss}")
            
        return W

    def setup_process(rank, world_size, backend='nccl'):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '3090'
        dist.init_process_group(backend, rank=rank, world_size=world_size)

    def cleanup():
        dist.destroy_process_group()

    def run_solve_opt_problem(rank, world_size, X, Y, batch_rows_length, is_y, n, kappa, plot_loss, return_dict):
        """
        ADMM Solve Wrapper.
        """
        setup_process(rank, world_size)
        
        # Set the correct device for this process
        gpu_id = self.gpu_ids[rank]
        torch.cuda.set_device(gpu_id)
        if isinstance(gpu_id, int):
            device = torch.device(f"cuda:{gpu_id}")
        else:
            device = gpu_id

        if is_y:
            num_epoch = self.num_epoch_cd
            rho = self.rho_cd
            lambda_yz = self.lambda_y
        else:
            num_epoch = self.num_epoch_ab
            rho = self.rho_ab
            lambda_yz = self.lambda_z

        Y = Y[:,rank*batch_rows_length:(rank+1)*batch_rows_length]

        if is_y:
            admm = ADMM_CD(X.shape[1], Y.shape[1], rho, lambda_yz, device=device)
        else:
            admm = ADMM_AB(X.shape[1], Y.shape[1], n, rho, lambda_yz, kappa, device=device)

        losses = []
        progress_bar = tqdm(
            range(num_epoch),
            desc="Training Epochs",
            disable=True if rank != 0 else False,
        )
        with torch.no_grad():
            for i in progress_bar:
                admm.step(X, Y)
                loss = admm.LassoObjective(X, Y)
                losses.append(loss)
                progress_bar.update(1)
                progress_bar.set_postfix({"Loss": loss})

        # Plot losses 
        if plot_loss:
            plt.figure()
            plt.plot(losses)
            plt.xlabel("Epoch")
            plt.ylabel("Lasso Objective")
            plt.yscale("log")
            plt.title(f"{rank} Training Loss")
            plt.savefig(f"loss_{rank}_isy_{is_y}.png")

            # Save the loss trace
            np.save(f"loss_trace_{rank}_isy_{is_y}.npy", losses)

        if is_y:
            result = admm.X.T.clone().detach().cpu().numpy()
        else:
            result = admm.avg.T.clone().detach().cpu().numpy()

        result[np.abs(result) <= config.sim.tol] = 0

        # Store the result in the shared dictionary
        return_dict[rank] = (result, losses[-1])
        
        cleanup()


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
        
        self.nu_X = torch.zeros(self.D, self.Q, device=device, requires_grad=False)
        self.nu_Z = torch.zeros(self.D, self.Q, device=device, requires_grad=False)
        self.nu_M = torch.zeros(self.D, self.Q, device=device, requires_grad=False)

        self.rho = rho

        self.X = torch.randn(self.D, self.Q, device=device, requires_grad=False)
        self.Z = torch.zeros(self.D, self.Q, device=device, requires_grad=False)
        self.M = torch.zeros(self.D, self.Q, device=device, requires_grad=False)
        self.avg = torch.zeros(self.D, self.Q, device=device, requires_grad=False)

        self.lambda_yz = lambda_yz
        self.kappa = kappa

    @torch.no_grad()
    def step(self, A, b):
        """
        ADMM's update step.
        """
        A = A.to(self.device)
        b = b.to(self.device)

        t1 = A.T.matmul(A) + self.rho * torch.eye(self.D, device=self.device)
        t2 = A.T.matmul(b) + self.rho * (self.avg - self.nu_X)
        self.X = torch.linalg.solve(t1, t2)

        self.Z = torch.sign(self.avg - self.nu_Z) * torch.clamp(torch.abs(self.avg - self.nu_Z) - self.lambda_yz / self.rho, min=0)

        self.M[:self.n,:] = self.project_w((self.avg - self.nu_M)[:self.n,:].T).T
        self.M[self.n:,:] = (self.avg - self.nu_M)[self.n:,:]

        self.avg = (self.X + self.Z + self.M) / 3

        self.nu_X = self.nu_X + (self.X - self.avg)
        self.nu_Z = self.nu_Z + (self.Z - self.avg)
        self.nu_M = self.nu_M + (self.M - self.avg)

    def project_w(self, matrix):
        """
        Project the matrix to the L1 ball.
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

        proj = torch.tensor(A_np, dtype=self.X.dtype, device=self.device)

        return proj

    @torch.no_grad()
    def LassoObjective(self, A, b):
        """
        Lasso objective function.
        """
        A = A.to(self.device)
        b = b.to(self.device)
        return (0.5 * torch.norm(A.matmul(self.avg) - b)**2 + self.lambda_yz * torch.sum(torch.abs(self.avg))).item()


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
        Solver class to solve C and D using consensus ADMM.

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
        self.X = torch.randn(self.D, self.Q, device=device)
        self.Z = torch.zeros(self.D, self.Q, device=device)
        self.lambda_yz = lambda_yz

    @torch.no_grad()
    def step(self, A, b):
        """
        ADMM's update step.
        """
        A = A.to(self.device)
        b = b.to(self.device)

        t1 = A.T.matmul(A) + self.rho * torch.eye(self.D, device=self.device)
        t2 = A.T.matmul(b) + self.rho * self.Z - self.nu
        self.X = torch.linalg.solve(t1, t2)

        self.Z = self.X + self.nu / self.rho - (self.lambda_yz / self.rho) * torch.sign(self.Z).to(self.device)
        self.nu = self.nu + self.rho * (self.X - self.Z)

    @torch.no_grad()
    def LassoObjective(self, A, b):
        """
        Lasso objective function.
        """
        A = A.to(self.device)
        b = b.to(self.device)
        return (0.5 * torch.norm(A.matmul(self.X) - b)**2 + self.lambda_yz * torch.sum(torch.abs(self.X))).item()
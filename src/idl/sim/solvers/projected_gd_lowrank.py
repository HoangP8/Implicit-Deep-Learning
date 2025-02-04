import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, Tuple, Optional

from ..utils import fixpoint_iteration

logger = logging.getLogger(__name__)

class LowRankModel(torch.nn.Module):
    def __init__(self, n, p, rank, kappa, device):
        """
        Low-rank model for solving A,B.
        """
        super(LowRankModel, self).__init__()
        self.device = device
        self.kappa = kappa

        if rank is None:
            rank = n
        elif rank > n:
            logger.warning("Rank is set to be larger than max rank, setting it back to full rank")
            rank = n

        self.L = nn.Parameter(torch.randn(n, rank, device=device))
        self.R = nn.Parameter(torch.randn(n, rank, device=device))
        self.B = nn.Parameter(torch.randn(n, p, device=device))

    def forward(self, X, U):
        """
        Forward pass with low-rank decomposition components.
        """
        X = X.to(self.device)
        U = U.to(self.device)
        output = self.L @ (self.R.T @ X) + self.B @ U
        return output

    def project_LR(self):
        """
        Project the low-rank components to satisfy the well-posedness condition.
        """
        self.L.data = self.project_w(self.L, self.kappa)
        self.R.data = self.project_w(self.R.T, self.kappa).T

    def project_w(self, matrix, v=0.99):
        """
        Project the matrix to the l1 norm ball.
        """
        A_np = matrix.detach().clone().cpu().numpy()
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

        proj = torch.tensor(A_np, dtype=matrix.dtype, device=matrix.device)

        return proj

class Trainer:
    def __init__(self, num_epoch, lambda_z, lr, verbose_epoch):
        """
        Initialize the trainer.
        """
        self.num_epoch = num_epoch
        self.lamb = lambda_z
        self.lr = lr
        self.verbose_epoch = verbose_epoch

    def train(self, model, X, U, Z):
        """
        Train the model.
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        losses = []
        for epoch in tqdm(range(self.num_epoch)):
            optimizer.zero_grad()
            output = model(X, U)
            loss = F.mse_loss(output, Z.to(output.device)) + self.lamb * torch.norm(model.L, p=2) + self.lamb * torch.norm(model.R, p=2)
            loss.backward()
            optimizer.step()
            model.project_LR()
            losses.append(loss.item())
            if epoch % self.verbose_epoch == 0:
                logger.info(f"Loss at epoch {epoch}: {loss.item()}")
        return model, losses

class ProjectedGDLowrankSolver:
    def __init__(
        self,
        rank : Optional[int] = None,
        num_epoch : int = 10000,
        lambda_z : float = 1e-6,
        lr : float = 1e-3,
        verbose_epoch : int = 100,
        regen_states : bool = False,
        tol : float = 1e-6,
    ):
        """
        Train State-driven Implicit Model using projected gradient descent to force A low-rank and well-posed.
        A, B are solved using projected gradient descent.
        C, D are solved using numpy least square solver.

        Args:
            rank (int): Rank of the A.
            num_epoch (int): Number of epochs to train A, B.
            lambda_z (float): Lasso regularization parameter for Z.
            lr (float): Learning rate.
            verbose_epoch (int): Number of epochs to print the loss.
            regen_states (bool, optional): Whether to regenerate states. Defaults to False.
            tol (float, optional): Tolerance for zeroing out weights. Defaults to 1e-6.
        """
        self.rank = rank
        self.num_epoch = num_epoch
        self.lambda_z = lambda_z
        self.lr = lr
        self.verbose_epoch = verbose_epoch
        self.regen_states = regen_states
        self.tol = tol

    def solve(
        self,
        X : np.ndarray,
        U : np.ndarray,
        Z : np.ndarray,
        Y : np.ndarray,
        model_config : Dict[str, Any],
        plot_loss : bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve the implicit model and force A to be low-rank.
        
        Args:
            X (np.ndarray): Post-activation array.
            U (np.ndarray): Input array.
            Z (np.ndarray): Pre-activation array.
            Y (np.ndarray): Output array.
            model_config (Dict[str, Any]): Implicit model configuration.
                - activation_fn (Callable): Activation function used by the implicit model.
                - device (str): Implicit model's device.
                - atol (float): Equilibrium function's tolerance.
                - kappa (float): Wellposedness condition parameter.
            plot_loss (bool, optional): Whether to plot the loss. Defaults to False.
        
        Returns:
            A, B, C, D (np.ndarray): Implicit model's parameters.
        """
        n, m, p, q = X.shape[0], X.shape[1], U.shape[0], Y.shape[0]

        X, U, Z = torch.tensor(X), torch.tensor(U), torch.tensor(Z)

        logger.info("===== Start parallel solve for A and B =====")
        L, R, B = self.lowrank_solve_matrix(X, U, Z, n=n, p=p, kappa=model_config["kappa"], device=model_config["device"], plot_loss=plot_loss)
        A = L @ R.T
        logger.info(f"Rank A: {np.linalg.matrix_rank(A)}")
        logger.info(f"Rank B: {np.linalg.matrix_rank(B)}")

        if self.regen_states:
            X = fixpoint_iteration(A, B, U, model_config["activation_fn"], model_config["device"], atol=model_config["atol"]).cpu()

        logger.info("===== Start parallel solve for C and D =====")
        CD = self.state_matching(np.hstack([X.cpu().numpy().T, U.cpu().numpy().T]), Y.T)
        C = CD[:, :n]
        D = CD[:, n:]

        return A, B, C, D

    def state_matching(self, X, Y):
        """
        Solve the least square problem with numpy.linalg.lstsq.
        """
        W, c, r, _ = np.linalg.lstsq(X, Y, rcond=None)

        loss = np.mean(np.square(X @ W - Y))
        
        logger.info(f"Total Lasso loss: {loss}")
        logger.info(f"Data rank: {r}")

        W[np.abs(W) <= self.tol] = 0
        
        return W.T

    def lowrank_solve_matrix(self, X, U, Z, n, p, kappa, device, plot_loss):
        """
        Solve the state matching problem with projected gradient descent to ensure A is low-rank and well-posed.
        """
        model = LowRankModel(n=n, p=p, rank=self.rank, kappa=kappa, device=device)
        trainer = Trainer(num_epoch=self.num_epoch, lambda_z=self.lambda_z, lr=self.lr, verbose_epoch=self.verbose_epoch)

        model, losses = trainer.train(model, X, U, Z)

        # Plot losses 
        if plot_loss:
            plt.figure()
            plt.plot(losses)
            plt.xlabel("Epoch")
            plt.ylabel("MSE loss")
            plt.yscale("log")
            plt.title(f"Training Loss")
            plt.savefig(f"loss_AB.png")

            # Save the loss trace
            np.save(f"loss_AB_trace.npy", losses)

        L = model.L.clone().detach().cpu().numpy()
        R = model.R.clone().detach().cpu().numpy()
        B = model.B.clone().detach().cpu().numpy()

        logger.info(f"Total loss: {losses[-1]}")

        return L, R, B


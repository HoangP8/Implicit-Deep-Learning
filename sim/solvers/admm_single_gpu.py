import gc
import logging
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os

from sim.utils import fixpoint_iteration, plot_heatmaps_and_histograms

logger = logging.getLogger(__name__)


def parallel_solve(X, U, Z, Y, config):
    n, m, p, q = X.shape[0], X.shape[1], U.shape[0], Y.shape[0]

    X, U, Z, Y = torch.tensor(X), torch.tensor(U), torch.tensor(Z), torch.tensor(Y)

    logger.info("===== Start parallel solve for A and B =====")
    AB = parallel_solve_matrix(torch.hstack([X.T, U.T]), Z.T, is_y=False, n=n, config=config)
    A = AB[:, :n]
    B = AB[:, n:]

    if config.sim.regen_states:
        X = fixpoint_iteration(A, B, U, config.activation, config.device, atol=config.sim.atol).cpu()

    if config.sim.solve_cd:
        logger.info("===== Start parallel solve for C and D =====")
        CD = parallel_solve_matrix(torch.hstack([X.T, U.T]), Y.T, is_y=True, n=n, config=config)
        C = CD[:, :n]
        D = CD[:, n:]
    else:
        C, D = None, None

    return A, B, C, D


def parallel_solve_matrix(X, Y, is_y, n, config):
    device = torch.device(config.device)
    total_rows = Y.shape[1]
    batch_rows_length = config.admm.batch_feature_size
    num_batches = total_rows // (batch_rows_length) + 1

    W = None
    loss = 0.0
    for k in range(num_batches):

        logger.info(f"Solving batch feature {k+1}/{num_batches}")

        start_idx = k * batch_rows_length
        end_idx = min((k + 1) * batch_rows_length, total_rows)
        Y_batch = Y[:, start_idx:end_idx]
        
        W_k, loss_k = run_solve_opt_problem(X, Y_batch, is_y, n, k, config, device)
        
        W = np.vstack([W, W_k]) if W is not None else W_k

        loss += loss_k
    
    logger.info(f"Total Lasso loss: {loss}")
    
    return W


def run_solve_opt_problem(X, Y, is_y, n, k, config, device):
    if is_y:
        num_epoch = config.gd.num_epoch_cd
        rho = config.admm.rho_cd
        lambda_yz = config.sim.lambda_y
        tau = config.admm.tau_cd
    else:
        num_epoch = config.gd.num_epoch_ab
        rho = config.admm.rho_ab
        lambda_yz = config.sim.lambda_z
        tau = config.admm.tau_ab

    if is_y:
        admm = ADMM_CD(X.shape[1], Y.shape[1], rho, lambda_yz, tau=tau, device=device)
    else:
        admm = ADMM_AB(X.shape[1], Y.shape[1], n, rho, lambda_yz, config.sim.kappa, tau=tau, device=device)

    losses = []
    with torch.no_grad():
        for i in tqdm(range(num_epoch)):
            admm.step(X, Y)
            
            # admm.rho = rho * (config.admm.rho_schedule) ** i
        
            losses.append(admm.LassoObjective(X, Y))

    # Plot losses 
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
        result = admm.X.T.clone().detach().cpu().numpy()
    else:
        result = admm.avg.T.clone().detach().cpu().numpy()

    result[np.abs(result) <= config.sim.tol] = 0

    return result, losses[-1]


class ADMM_AB:
    def __init__(self, D, Q, n, rho, lambda_yz, kappa, tau, device):
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
        self.tau = tau

    @torch.no_grad()
    def step(self, A, b):
        A = A.to(self.device)
        b = b.to(self.device)

        t1 = A.T.matmul(A) + self.rho * torch.eye(self.D, device=self.device)
        t2 = A.T.matmul(b) + self.rho * (self.avg - self.nu_X)
        self.X = torch.linalg.solve(t1, t2)

        self.Z = torch.sign(self.avg - self.nu_Z) * torch.clamp(torch.abs(self.avg - self.nu_Z) - self.lambda_yz / self.rho, min=0)

        self.M[:self.n,:] = self.project_w((self.avg - self.nu_M)[:self.n,:].T).T
        self.M[self.n:,:] = (self.avg - self.nu_M)[self.n:,:]

        self.avg = (self.X + self.Z + self.M) / 3

        self.nu_X = self.nu_X + self.tau * (self.X - self.avg)
        self.nu_Z = self.nu_Z + self.tau * (self.Z - self.avg)
        self.nu_M = self.nu_M + self.tau * (self.M - self.avg)

    def project_w(self, matrix):
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
        A = A.to(self.device)
        b = b.to(self.device)
        return (0.5 * torch.norm(A.matmul(self.avg) - b)**2 + self.lambda_yz * torch.sum(torch.abs(self.avg))).item()


class ADMM_CD:
    def __init__(self, D, Q, rho, lambda_yz, tau, device):
        self.D = D
        self.Q = Q
        self.device = device
        
        self.nu = torch.zeros(self.D, self.Q, device=device)
        self.rho = rho
        self.X = torch.randn(self.D, self.Q, device=device)
        self.Z = torch.zeros(self.D, self.Q, device=device)
        self.lambda_yz = lambda_yz
        self.tau = tau

    @torch.no_grad()
    def step(self, A, b):
        A = A.to(self.device)
        b = b.to(self.device)

        t1 = A.T.matmul(A) + self.rho * torch.eye(self.D, device=self.device)
        t2 = A.T.matmul(b) + self.rho * self.Z - self.nu
        self.X = torch.linalg.solve(t1, t2)

        self.Z = self.X + self.nu / self.rho - (self.lambda_yz / self.rho) * torch.sign(self.Z).to(self.device)
        self.nu = self.nu + self.rho * self.tau * (self.X - self.Z)

    @torch.no_grad()
    def LassoObjective(self, A, b):
        A = A.to(self.device)
        b = b.to(self.device)
        return (0.5 * torch.norm(A.matmul(self.X) - b)**2 + self.lambda_yz * torch.sum(torch.abs(self.X))).item()
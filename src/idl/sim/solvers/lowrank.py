import gc
import logging
import torch
import torch.nn as nn
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

    X, U, Z = torch.tensor(X), torch.tensor(U), torch.tensor(Z)

    logger.info("===== Start parallel solve for A and B =====")
    L, R, B = lowrank_solve_matrix(X, U, Z, n=n, p=p, config=config)
    A = L @ R.T
    logger.info(f"Rank A: {np.linalg.matrix_rank(A)}")
    logger.info(f"Rank B: {np.linalg.matrix_rank(B)}")

    if config.sim.regen_states:
        X = fixpoint_iteration(A, B, U, config.activation, config.device, atol=config.sim.atol).cpu()

    logger.info("===== Start parallel solve for C and D =====")
    CD = state_matching(np.hstack([X.cpu().numpy().T, U.cpu().numpy().T]), Y.T, config=config)
    C = CD[:, :n]
    D = CD[:, n:]

    return A, B, C, D

def state_matching(X, Y, config):
    
    W, c, r, _ = np.linalg.lstsq(X, Y, rcond=None)

    loss = np.mean(np.square(X @ W - Y))
    
    logger.info(f"Total Lasso loss: {loss}")
    logger.info(f"Data rank: {r}")

    W[np.abs(W) <= config.sim.tol] = 0
    
    return W.T

def lowrank_solve_matrix(X, U, Z, n, p, config):
    gpu_id = config.gd.gpu_list[0]
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    logger.info(f"Device set to: {device}")

    model = LowRankModel(n=n, p=p, config=config, device=device)
    trainer = Trainer(config=config)

    model, losses = trainer.train(model, X, U, Z)

    # Plot losses 
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

class LowRankModel(torch.nn.Module):
    def __init__(self, n, p, config, device):
        super(LowRankModel, self).__init__()
        self.device = device
        self.kappa = config.sim.kappa
        k = int(n * config.lowrank.rank_rate)
        logger.info(f"Compressed dim k: {k}")
        self.L = nn.Parameter(torch.randn(n, k, device=device))
        self.R = nn.Parameter(torch.randn(n, k, device=device))
        self.B = nn.Parameter(torch.randn(n, p, device=device))
        if config.lowrank.with_identity:
            raise NotImplementedError()

    def forward(self, X, U):
        X = X.to(self.device)
        U = U.to(self.device)
        output = self.L @ (self.R.T @ X) + self.B @ U
        return output

    def project_LR(self):
        self.L.data = self.project_w(self.L, self.kappa)
        self.R.data = self.project_w(self.R.T, self.kappa).T

    def project_w(self, matrix, v=0.99):
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
    def __init__(self, config):
        self.num_epoch = config.gd.num_epoch_ab
        self.lamb = config.sim.lambda_z
        self.batch_size = config.gd.batch_size
        self.lr = config.gd.lr_ab
        self.momentum = config.gd.momentum
        self.checkpoint_epoch = config.gd.checkpoint_epoch

    def train(self, model, X, U, Z):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        losses = []
        for epoch in tqdm(range(self.num_epoch)):
            optimizer.zero_grad()
            output = model(X, U)
            loss = F.mse_loss(output, Z.to(output.device))
            loss.backward()
            optimizer.step()
            model.project_LR()
            losses.append(loss.item())
            if epoch % self.checkpoint_epoch == 0:
                logger.info(f"Loss at epoch {epoch}: {loss.item()}")
        return model, losses


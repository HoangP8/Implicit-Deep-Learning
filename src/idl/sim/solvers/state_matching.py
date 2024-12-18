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

    # X, U, Z, Y = torch.tensor(X), torch.tensor(U), torch.tensor(Z), torch.tensor(Y)

    logger.info("===== Start parallel solve for A and B =====")
    AB = parallel_solve_matrix(np.hstack([X.T, U.T]), Z.T, is_y=False, n=n, config=config)
    A = AB[:, :n]
    B = AB[:, n:]

    if config.sim.regen_states:
        X = fixpoint_iteration(A, B, U, config.device, atol=config.sim.atol).cpu()

    logger.info("===== Start parallel solve for C and D =====")
    CD = parallel_solve_matrix(np.hstack([X.T, U.T]), Y.T, is_y=True, n=n, config=config)
    C = CD[:, :n]
    D = CD[:, n:]

    return A, B, C, D


def parallel_solve_matrix(X, Y, is_y, n, config):
    
    W, c, r, _ = np.linalg.lstsq(X, Y, rcond=None)

    loss = np.mean(np.square(X @ W - Y))
    
    logger.info(f"Total Lasso loss: {loss}")
    logger.info(f"Data rank: {r}")

    W[np.abs(W) <= config.sim.tol] = 0
    
    return W.T

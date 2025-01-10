import gc
import logging
import math
import time
import tracemalloc
from multiprocessing import Pool, shared_memory

import sim.solvers.mosek as cp
import numpy as np
from celer import ElasticNet, Lasso
from sklearn import linear_model
from scipy import sparse
from tqdm import tqdm

from sim.utils import fixpoint_iteration

logger = logging.getLogger(__name__)


def create_shared_memory_block(ndarray_to_share):
    # create a shared memory of size array.nbytes
    shm_blocks = [
        shared_memory.SharedMemory(create=True, size=array.nbytes)
        for array in ndarray_to_share
    ]
    # create a ndarray using the buffer of shm
    ndarray_shm = [
        np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
        for (array, shm) in zip(ndarray_to_share, shm_blocks)
    ]
    # copy the data into the shared memory
    for array_shm, array in zip(ndarray_shm, ndarray_to_share):
        array_shm[:] = array[:]
    return shm_blocks, ndarray_shm


def elastic_net_loss(yz, a, b, X, U, is_y, config):
    """Calculate the Elastic Net loss."""
    if is_y:
        lambda_yz = config.sim.lambda_y
    else:
        lambda_yz = config.sim.lambda_z
    # Calculate L1 regularization term
    l1_term = config.celer.l1_ratio * (np.sum(np.abs(a)) + np.sum(np.abs(b)))

    # Calculate L2 regularization term
    l2_term = (1 - config.celer.l1_ratio) * 0.5 * (np.sum(np.square(a)) + np.sum(np.square(b)))

    # Calculate the mean squared error (MSE)
    mse = np.sum(np.square(yz - (np.hstack([X.T, U.T]) @ np.vstack([a, b]))))

    # Calculate the total loss
    loss = 1/(2 * X.shape[1]) * mse + lambda_yz * (l1_term + l2_term)

    return loss

def lasso_loss(yz, a, b, X, U, is_y, config):
    """Calculate the Lasso loss."""
    if is_y:
        lambda_yz = config.sim.lambda_y
    else:
        lambda_yz = config.sim.lambda_z
    # Calculate L1 regularization term
    l1_term = np.sum(np.abs(a)) + np.sum(np.abs(b))

    # Calculate the mean squared error (MSE)
    mse = np.sum(np.square(yz - (np.hstack([X.T, U.T]) @ np.vstack([a, b]))))

    # Calculate the total loss
    # loss = 1/(2 * X.shape[1]) * mse + lambda_yz * l1_term
    loss = 0.5 * mse + lambda_yz * l1_term

    return loss

def get_objective_cvxpy(a, b, yz_param, X, U, is_y, config):
    """Return objective and constraints for sequential solver."""
    # Note: robust optimizer formula: minimize vectorized L1 min(sim|M_i,j|)

    objective = 0
    constraints = []
    if is_y:
        lambda_yz = config.sim.lambda_y
    else:
        lambda_yz = config.sim.lambda_z

    # Elastic Net
    if config.sim.elastic_net:
        objective += lambda_yz * config.celer.l1_ratio * (cp.pnorm(a, 1) + cp.pnorm(b, 1))
        objective += 0.5 * lambda_yz * (1 - config.celer.l1_ratio) * (cp.pnorm(a, 2)**2 + cp.pnorm(b, 2)**2)
    else:
        # L1 objective
        objective = objective + lambda_yz * (cp.pnorm(a, 1) + cp.pnorm(b, 1))

    if config.sim.regularized:
        objective = (
            objective
            # + 1/(2 * X.shape[1]) * cp.pnorm(yz_param - (np.hstack([X.T, U.T]) @ cp.vstack([a, b])), 2) ** 2
            + 0.5 * cp.pnorm(yz_param - (np.hstack([X.T, U.T]) @ cp.vstack([a, b])), 2) ** 2
        )
    else:
        # exact matching constraint
        constraints.append((X.T @ a) + (U.T @ b) == yz_param)

    # well-posedness constraint
    if not is_y and config.sim.well_pose:
        constraints.append(cp.pnorm(a, 1) <= config.sim.kappa)

    return cp.Minimize(objective), constraints


def solve_opt_problem_cvxpy(X, U, yz, is_y, problem_size, config):
    n, m, p, _ = problem_size

    # variables for model weights
    a = cp.Variable((n, config.sim.num_row))
    b = cp.Variable((p, config.sim.num_row))

    # set up parameters
    yz_param = cp.Parameter((m, config.sim.num_row))

    objective, constraints = get_objective_cvxpy(
        a,
        b,
        yz_param,
        X,
        U,
        is_y,
        config,
    )
    yz_param.value = yz
    prob = cp.Problem(objective, constraints)

    # try:
    prob.solve(verbose=False, solver=cp.MOSEK)
    # except:
    # prob.solve(verbose=False, solver=cp.ECOS)
    # prob.solve(verbose=False, solver=cp.CLARABEL)

    # threshold entries to enforce exact zeros and store array in compress sparse row format
    a.value[abs(a.value) <= config.sim.tol] = 0
    b.value[abs(b.value) <= config.sim.tol] = 0
    # logger.info(f"1-norm of a: {np.linalg.norm(a.value, 1)}.")

    if config.sim.elastic_net:
        loss = elastic_net_loss(yz, a.value, b.value, X, U, is_y, config)
        # logger.info(f"ElasticNet loss: {elastic_loss}")
    elif config.sim.regularized:
        loss = lasso_loss(yz, a.value, b.value, X, U, is_y, config)
    else:
        loss = 0

    a_sp = sparse.csr_matrix(a.value.T)
    b_sp = sparse.csr_matrix(b.value.T)

    return a_sp, b_sp, loss


def solve_opt_problem(X_shm_name, U_shm_name, yz, is_y, problem_size, config):
    n, m, p, _ = problem_size

    X_shm = shared_memory.SharedMemory(name=X_shm_name)
    U_shm = shared_memory.SharedMemory(name=U_shm_name)
    X = np.ndarray((n, m), dtype="float32", buffer=X_shm.buf)
    U = np.ndarray((p, m), dtype="float32", buffer=U_shm.buf)

    return solve_opt_problem_cvxpy(X, U, yz, is_y, problem_size, config)


def parallel_solve_matrix(X_shm, U_shm, YZ, is_y, problem_size, config):
    # initialize empty list to store csr format sparse matrix
    A, B = None, None
    loss = []

    # start batch processing
    total_processes = math.ceil(YZ.shape[0] / config.sim.num_row)
    batch_size = config.sim.batch_size
    for batch in tqdm(range(0, YZ.shape[0], batch_size)):
        # construct parallel input data for a batch
        batch_end = min(batch + batch_size, YZ.shape[0])
        parallel_input = [
            (
                X_shm.name,
                U_shm.name,
                YZ[i : min(i + config.sim.num_row, YZ.shape[0])].T,
                is_y,
                problem_size,
                config,
            )
            for i in range(batch, batch_end, config.sim.num_row)
        ]
        # logger.info(f"Solving batch {batch}")

        # construct cvxpy with multiprocessing
        with Pool(processes=config.sim.processes) as pool:
            results = []

            for result in pool.starmap(solve_opt_problem, parallel_input):
                results.append(result)

        # unzip a list of tuples
        A_lst, B_lst, losses = list(zip(*results))
        # store in scipy csr_matrix format
        A_sp, B_sp = sparse.vstack(A_lst), sparse.vstack(B_lst)
        A = sparse.vstack([A, A_sp]) if A is not None else A_sp
        B = sparse.vstack([B, B_sp]) if B is not None else B_sp
        del A_lst, B_lst, A_sp, B_sp  # free memory
        gc.collect()

        loss.extend(losses)

    loss = np.sum(loss)
    logger.info(f"Total loss: {loss}")

    return A, B


def mosek_solver(X, U, Z, Y, config):
    """
    Wrapper for using mosek.
    """
    n, m, p, q = X.shape[0], X.shape[1], U.shape[0], Y.shape[0]

    (X_shm, U_shm), (X, U) = create_shared_memory_block([X, U])

    logger.info("===== Start parallel solve for A and B =====")
    A, B = parallel_solve_matrix(X_shm, U_shm, Z, False, (n, m, p, q), config)

    if config.sim.regen_states:
        X = fixpoint_iteration(A, B, U, config.device).cpu().numpy()
        (X_shm, U_shm), (X, U) = create_shared_memory_block([X, U])

    if config.sim.solve_cd: 
        logger.info("===== Start parallel solve for C and D =====")
        C, D = parallel_solve_matrix(X_shm, U_shm, Y, True, (n, m, p, q), config)
    else:
        C, D = None, None

    X_shm.close()
    X_shm.unlink()
    U_shm.close()
    U_shm.unlink()

    return A, B, C, D
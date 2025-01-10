import gc
import logging
import math
import time
from multiprocessing import Pool, shared_memory

import cvxpy as cp
import numpy as np
from scipy import sparse
from tqdm import tqdm

from ..utils import fixpoint_iteration

logger = logging.getLogger(__name__)

def create_shared_memory_block(ndarray_to_share):
    """
    Create a shared memory block for the given ndarray to be used by multiple processes.
    """
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

class CVXSolver:
    def __init__(
        self,
        num_row : int = 1,
        batch_size : int = 32,
        num_processes : int = 32,
        lambda_y : float = 1e-6,
        lambda_z : float = 1e-6,
        l1_ratio : float = 0.5,
        elastic_net : bool = False,
        regularized : bool = True,
        well_pose : bool = True,
        regen_states : bool = False,
        tol : float = 1e-6,
    ):
        """
        Train State-driven Implicit Model using CVXPY.
        This is highly recommended for small-scale problems due to its high precision and easy tuning.

        Args:
            num_row (int, optional): Number of rows to solve in each process each iteration. Defaults to 1.
            batch_size (int, optional): Number of columns to solve in each solving iteration. Defaults to 32.
            num_processes (int, optional): Number of processes available. Defaults to 32.
            lambda_y (float, optional): Lasso regularization parameter for Y. Defaults to 1e-6.
            lambda_z (float, optional): Lasso regularization parameter for Z. Defaults to 1e-6.
            l1_ratio (float, optional): Elastic Net regularization parameter. Defaults to 0.5.
            elastic_net (bool, optional): Whether to use Elastic Net regularization. Defaults to False.
            regularized (bool, optional): Whether to use L1 regularization. Defaults to True.
            well_pose (bool, optional): Whether to enforce well-posedness constraint. Defaults to True.
            regen_states (bool, optional): Whether to regenerate states. Defaults to False.
            tol (float, optional): Tolerance for zeroing out weights. Defaults to 1e-6.
        """
        self.num_row = num_row
        self.batch_size = batch_size
        self.processes = num_processes
        self.lambda_z = lambda_z
        self.lambda_y = lambda_y
        self.l1_ratio = l1_ratio
        self.elastic_net = elastic_net
        self.regularized = regularized
        self.well_pose = well_pose
        self.regen_states = regen_states
        self.tol = tol

    def solve(
        self, 
        X : np.ndarray, 
        U : np.ndarray, 
        Z : np.ndarray, 
        Y : np.ndarray, 
        model_config : Dict[str, Any]
    ):
        """
        Solve State-driven Implicit Model using CVXPY.

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

        (X_shm, U_shm), (X, U) = create_shared_memory_block([X, U])

        logger.info("===== Start parallel solve for A and B =====")
        A, B = self.parallel_solve_matrix(X_shm, U_shm, Z, False, (n, m, p, q), model_config["kappa"])

        if self.regen_states:
            X = fixpoint_iteration(A, B, U, model_config["activation_fn"], model_config["device"], atol=model_config["atol"]).cpu().numpy()
            (X_shm, U_shm), (X, U) = create_shared_memory_block([X, U])

        logger.info("===== Start parallel solve for C and D =====")
        C, D = self.parallel_solve_matrix(X_shm, U_shm, Y, True, (n, m, p, q), model_config["kappa"])

        X_shm.close()
        X_shm.unlink()
        U_shm.close()
        U_shm.unlink()

        return A, B, C, D

    def parallel_solve_matrix(self, X_shm, U_shm, YZ, is_y, problem_size, kappa):
        """
        Dividing the data matrix and solve parallelly on CPU over multiple processes.
        """
        # initialize empty list to store csr format sparse matrix
        A, B = None, None
        loss = []

        # start batch processing
        total_processes = math.ceil(YZ.shape[0] / self.num_row)
        for batch in tqdm(range(0, YZ.shape[0], self.batch_size)):
            # construct parallel input data for a batch
            batch_end = min(batch + self.batch_size, YZ.shape[0])
            parallel_input = [
                (
                    X_shm.name,
                    U_shm.name,
                    YZ[i : min(i + self.num_row, YZ.shape[0])].T,
                    is_y,
                    problem_size,
                    kappa,
                )
                for i in range(batch, batch_end, self.num_row)
            ]
            # logger.info(f"Solving batch {batch}")

            # construct cvxpy with multiprocessing
            with Pool(processes=self.processes) as pool:
                results = []

                for result in pool.starmap(self.solve_opt_problem, parallel_input):
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

    def solve_opt_problem(self, X_shm_name, U_shm_name, yz, is_y, problem_size, kappa):
        """
        Solve the optimization problem using CVXPY.
        """
        n, m, p, _ = problem_size

        X_shm = shared_memory.SharedMemory(name=X_shm_name)
        U_shm = shared_memory.SharedMemory(name=U_shm_name)
        X = np.ndarray((n, m), dtype="float32", buffer=X_shm.buf)
        U = np.ndarray((p, m), dtype="float32", buffer=U_shm.buf)

        # variables for model weights
        a = cp.Variable((n, self.num_row))
        b = cp.Variable((p, self.num_row))

        # set up parameters
        yz_param = cp.Parameter((m, self.num_row))

        objective, constraints = self.get_objective_cvxpy(
            a,
            b,
            yz_param,
            X,
            U,
            is_y,
            kappa,
        )
        yz_param.value = yz
        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(verbose=False, solver=cp.MOSEK)
        except:
            # prob.solve(verbose=False, solver=cp.ECOS)
            prob.solve(verbose=False, solver=cp.CLARABEL)

        # threshold entries to enforce exact zeros and store array in compress sparse row format
        a.value[abs(a.value) <= self.tol] = 0
        b.value[abs(b.value) <= self.tol] = 0
        # logger.info(f"1-norm of a: {np.linalg.norm(a.value, 1)}.")

        if self.elastic_net:
            loss = self.elastic_net_loss(yz, a.value, b.value, X, U, is_y)
            # logger.info(f"ElasticNet loss: {elastic_loss}")
        elif self.regularized:
            loss = self.lasso_loss(yz, a.value, b.value, X, U, is_y)
        else:
            loss = 0

        a_sp = sparse.csr_matrix(a.value.T)
        b_sp = sparse.csr_matrix(b.value.T)

        return a_sp, b_sp, loss

    def get_objective_cvxpy(self, a, b, yz_param, X, U, is_y, kappa):
        """
        Return objective and constraints for sequential solver.
        Note: robust optimizer formula: minimize vectorized L1 min(sim|M_i,j|)
        """
        objective = 0
        constraints = []
        if is_y:
            lambda_yz = self.lambda_y
        else:
            lambda_yz = self.lambda_z

        if self.elastic_net:
            # Elastic Net
            objective += lambda_yz * self.l1_ratio * (cp.pnorm(a, 1) + cp.pnorm(b, 1))
            objective += 0.5 * lambda_yz * (1 - self.l1_ratio) * (cp.pnorm(a, 2)**2 + cp.pnorm(b, 2)**2)
        else:
            # L1 objective
            objective = objective + lambda_yz * (cp.pnorm(a, 1) + cp.pnorm(b, 1))

        if self.regularized:
            objective = (
                objective
                # + 1/(2 * X.shape[1]) * cp.pnorm(yz_param - (np.hstack([X.T, U.T]) @ cp.vstack([a, b])), 2) ** 2
                + 0.5 * cp.pnorm(yz_param - (np.hstack([X.T, U.T]) @ cp.vstack([a, b])), 2) ** 2
            )
        else:
            # exact matching constraint
            constraints.append((X.T @ a) + (U.T @ b) == yz_param)

        # well-posedness constraint
        if not is_y and self.well_pose:
            constraints.append(cp.pnorm(a, 1) <= kappa)

        return cp.Minimize(objective), constraints

    def elastic_net_loss(self, yz, a, b, X, U, is_y):
        """Calculate the Elastic Net loss."""
        if is_y:
            lambda_yz = self.lambda_y
        else:
            lambda_yz = self.lambda_z
        # Calculate L1 regularization term
        l1_term = self.l1_ratio * (np.sum(np.abs(a)) + np.sum(np.abs(b)))

        # Calculate L2 regularization term
        l2_term = (1 - self.l1_ratio) * 0.5 * (np.sum(np.square(a)) + np.sum(np.square(b)))

        # Calculate the mean squared error (MSE)
        mse = np.sum(np.square(yz - (np.hstack([X.T, U.T]) @ np.vstack([a, b]))))

        # Calculate the total loss
        loss = 1/(2 * X.shape[1]) * mse + lambda_yz * (l1_term + l2_term)

        return loss

    def lasso_loss(self, yz, a, b, X, U, is_y):
        """Calculate the Lasso loss."""
        if is_y:
            lambda_yz = self.lambda_y
        else:
            lambda_yz = self.lambda_z
        # Calculate L1 regularization term
        l1_term = np.sum(np.abs(a)) + np.sum(np.abs(b))

        # Calculate the mean squared error (MSE)
        mse = np.sum(np.square(yz - (np.hstack([X.T, U.T]) @ np.vstack([a, b]))))

        # Calculate the total loss
        # loss = 1/(2 * X.shape[1]) * mse + lambda_yz * l1_term
        loss = 0.5 * mse + lambda_yz * l1_term

        return loss


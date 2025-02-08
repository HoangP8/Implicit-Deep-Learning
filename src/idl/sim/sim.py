import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler

from contextlib import contextmanager
import logging
from typing import Callable, Dict, List, Optional, Union

# from .solvers import solve
from .utils import fixpoint_iteration

logger = logging.getLogger(__name__)

__activation_types__ = (
    nn.CELU,
    nn.ELU,
    nn.GELU,
    nn.GLU,
    nn.Hardshrink,
    nn.Hardsigmoid,
    nn.Hardswish,
    nn.Hardtanh,
    nn.LeakyReLU,
    nn.LogSigmoid,
    nn.LogSoftmax,
    nn.Mish,
    nn.MultiheadAttention,
    nn.PReLU,
    nn.ReLU,
    nn.ReLU6,
    nn.RReLU,
    nn.SELU,
    nn.Sigmoid,
    nn.SiLU,
    nn.Softmax,
    nn.Softmax2d,
    nn.Softmin,
    nn.Softplus,
    nn.Softshrink,
    nn.Softsign,
    nn.Tanh,
    nn.Tanhshrink,
    nn.Threshold,
)

class HookManager:
    def __init__(self):
        """
        HookManager class to collect pre- and post-activations from a model.
        """
        self.pre_activations = []
        self.post_activations = []
        self.hooks = []

    def hook_fn(self, module : torch.nn.Module, input : torch.Tensor, output : torch.Tensor):
        """
        Hook function to collect pre- and post-activations from a model.
        """
        if input[0].size() == output.size():
            self.pre_activations.append(input[0].detach())
            self.post_activations.append(output.detach())

    def _apply_all_hooks(self, module : torch.nn.Module):
        """
        Recursively apply hooks to all activation layers in the model.
        """
        children = list(module.children())
        if len(children) == 0:  # It's a leaf module
            if isinstance(module, __activation_types__): # check if the module is an activation layer
                hook = module.register_forward_hook(self.hook_fn)
                self.hooks.append(hook)
        else:
            for child in module.children():
                self._apply_all_hooks(child)  # Recursively apply to children
    
    def _check_modules(self, module : torch.nn.Module, list_relu : List[torch.nn.Module]):
        """
        Recursively look for activation layers inside the module.
        """
        children = list(module.children())
        if len(children) == 0:  # It's a leaf module
            if isinstance(module, __activation_types__):
                list_relu.append(module)
        else:
            for child in module.children():
                self._check_modules(child, list_relu)  # Recursively apply to children

    def _apply_n_hooks(self, module : torch.nn.Module, n : int = 3):
        """
        Skip some activations and apply hooks only to the last activation layer of each layer block.

        Args:
            module (torch.nn.Module): The module to apply hooks to.
            n (int): The threshold to define a block of layers. Only skip layers when the number of children is greater than n.
        """
        children_length = len(list(module.children()))
        if children_length == 0:
            if isinstance(module, __activation_types__):
                hook = module.register_forward_hook(self.hook_fn)
                self.hooks.append(hook)
        elif children_length <= n: # if the number of children is less than n, apply hooks to all children
            for child in module.children():
                self._apply_n_hooks(child, n)
        else: # if the number of children is greater than n, apply hooks to the last activation of the layer block
            for child in module.children():
                relu_layers = []
                self._check_modules(child, relu_layers)
                if relu_layers:
                    final_relu = relu_layers[-1] # get only the last activation of a block of layers
                    hook = final_relu.register_forward_hook(self.hook_fn)
                    self.hooks.append(hook)

    @contextmanager
    def register_hooks(self, model : torch.nn.Module, skip_layers : Optional[int] = None):
        """
        Register hooks to all activation layers in the model.
        """
        if skip_layers is None:
            self._apply_all_hooks(model)
        else:
            self._apply_n_hooks(model, skip_layers)
        try:
            yield
        finally:
            for hook in self.hooks:
                hook.remove()
            self.pre_activations.clear()
            self.post_activations.clear()
            self.hooks.clear()


class SIM():
    def __init__(
        self,
        activation_fn : Callable = nn.ReLU,
        kappa : float = 0.99,
        atol : float = 1e-6,
        skip_layers : Optional[int] = None,
        standardize : bool = False,
        device : Optional[Union[str, torch.device]] = None, 
        dtype : Optional[Union[str, torch.dtype]] = None,
    ):
        """
        SIM base class.

        Args:
            config (dict): Configuration dictionary.
            device (str or torch.device, optional): Device to use for SIM. Defaults to None.
            dtype (str or torch.dtype, optional): Data type for SIM. Defaults to None.
            standardize (bool, optional): Whether to standardize the input data using scipy StandardScaler. Defaults to False.
        """
        self.activation_fn = activation_fn
        self.atol = atol
        self.kappa = kappa
        self.skip_layers = skip_layers
        self.device = device
        self.dtype = dtype
        self.weights = {
            'A': None,
            'B': None,
            'C': None,
            'D': None,
        }
        self.standardize = standardize
        if standardize:
            self.scaler = StandardScaler()

    def to(
        self,
        device : Optional[Union[str, torch.device]] = None, 
        dtype : Optional[Union[str, torch.dtype]] = None
    ):
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        for weight in self.weights.keys():
            if self.weights[weight] is not None:
                self.weights[weight] = self.weights[weight].to(self.device, self.dtype)
        return self
    
    def __call__(
        self, 
        input : torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of a standard implicit model.

        Args:
            input (torch.Tensor): Input data with shape (batch_size, input_dim).
        
        Returns:
            output (torch.Tensor): Output data with shape (batch_size, output_dim).
        """
        for weight in self.weights.keys():
            assert self.weights[weight] is not None, f"Weight matrix {weight} is not trained"

        X = fixpoint_iteration(self.weights['A'], self.weights['B'], input, self.activation_fn, self.device, atol=self.atol)
        Y = X @ self.weights['C'] + self.weights['D'] @ input

        return Y
    
    def get_states(
            self, 
            model : torch.nn.Module, 
            inputs : torch.Tensor,
    ) -> Dict[str, np.ndarray]:
        """
        Extract the states data (pre-activations, post-activations, inputs, outputs) from the explicit model.
        The dataloader should only load a small amount of data to avoid memory issues.

        Args:
            model (torch.nn.Module): Explicit model (teacher) to extract state data.
            inputs (torch.Tensor): Input data with shape (batch_size, input_dim).
        
        Returns:
            states_data (dict): Dictionary containing the states data:
                - U (np.ndarray): Input data with shape (batch_size, input_dim).
                - Z (np.ndarray): Pre-activations with shape (batch_size, hidden_dim).
                - X (np.ndarray): Post-activations with shape (batch_size, hidden_dim).
                - Y (np.ndarray): Output data with shape (batch_size, output_dim).
        """
        model.requires_grad_(False)
        model.eval()
        model.to(self.device)
        
        hooks = HookManager()

        outputs_accumulated = []
        inputs_accumulated = []
        pre_activations_accumulated = []
        post_activations_accumulated = []

        for input_samples in inputs:
            input_samples = input_samples.to(self.device)

            # Run the model with hooks and no gradient calculations
            with hooks.register_hooks(model, skip_layers=self.skip_layers), torch.no_grad():
                outputs = model(input_samples)

                # Accumulate outputs
                outputs_accumulated.append(
                    outputs.cpu().numpy()
                )  # Convert to NumPy array and store

                # Construct and accumulate implicit representation
                U = input_samples.flatten(1).cpu().numpy()
                inputs_accumulated.append(U)

                # Accumulate activations
                num_layers = len(hooks.pre_activations)

                Z = np.hstack(
                    [
                        hooks.pre_activations[i].flatten(1).cpu().numpy()
                        for i in range(num_layers - 1, -1, -1)
                    ]
                )
                pre_activations_accumulated.append(Z)

                X = np.hstack(
                    [
                        hooks.post_activations[i].flatten(1).cpu().numpy()
                        for i in range(num_layers - 1, -1, -1)
                    ]
                )
                post_activations_accumulated.append(X)

        states_data = {}
        states_data['Y'] = np.vstack(outputs_accumulated).T
        states_data['U'] = np.vstack(inputs_accumulated).T
        states_data['Z'] = np.vstack(pre_activations_accumulated).T
        states_data['X'] = np.vstack(post_activations_accumulated).T

        if self.standardize:
            states_data['U'] = self.scaler.fit_transform(states_data['U'].T).T
            states_data['Z'] = self.scaler.fit_transform(states_data['Z'].T).T
            states_data['X'] = self.scaler.fit_transform(states_data['X'].T).T
            states_data['Y'] = self.scaler.fit_transform(states_data['Y'].T).T

        return states_data
    
    def train(
        self,
        solver : Callable,
        model : Optional[torch.nn.Module] = None,
        dataloader : Optional[torch.utils.data.DataLoader] = None,
        states_data_path : Optional[str] = None,
        save_states_path : Optional[str] = None,
    ):
        """
        Train the SIM model.

        Args:
            model (torch.nn.Module): Explicit model (teacher) to extract state data.
            dataloader (torch.utils.data.DataLoader): Training data loader.
            states_data_path (str, optional): Path to the states data file. Defaults to None.
            save_states_path (str, optional): Path to save the states data file. If provided, the extracted states data will be saved to this file. Defaults to None.
            solver (str, optional): Solver to use for training. Defaults to "mosek".
            regen_states (bool, optional): Whether to regenerate states data. Defaults to False.
            tol (float, optional): Set weights less than tol to 0. Defaults to 1e-6.
        """
        # check function arguments for states extraction
        if model is None and states_data_path is None:
            raise ValueError("Either model or states_data_path must be provided")
        elif model is not None and dataloader is None:
            raise ValueError("dataloader must be provided for states extraction")

        if states_data_path is not None and save_states_path is not None:
            logger.warning("Both states_data_path and save_states_path are provided. Only states_data_path will be used.")

        if states_data_path is None:
            logger.info("===== Collecting states =====")
            states_data = self.get_states(model, dataloader)
            if save_states_path is not None:
                logger.info(f"===== Saving states to {save_states_path} =====")
                np.save(save_states_path, states_data)
        else:
            logger.info(f"===== Loading states from {states_data_path} =====")
            states_data = np.load(states_data_path)

        logger.info("===== Start training SIM =====")

        training_config = {
            'activation_fn': self.activation_fn,
            'device': self.device,
            'atol': self.atol,
            'kappa': self.kappa,
        }

        A, B, C, D = solver.solve(states_data['X'], states_data['U'], states_data['Z'], states_data['Y'], training_config)

        logger.info(f"===== Training finished =====")

        self.weights['A'] = A
        self.weights['B'] = B
        self.weights['C'] = C
        self.weights['D'] = D

        return self
    
    def evaluate(
        self,
        inputs : torch.Tensor,
        labels : torch.Tensor,
    ) -> float:
        """
        Evaluate the SIM model on the given test data.

        Args:
            inputs (torch.Tensor): Input data with shape (batch_size, input_dim).
            labels (torch.Tensor): Labels for the input data with shape (batch_size,).
        
        Returns:
            test_accuracy (float): Accuracy of the SIM model on the given test data.
        """
        U_test = inputs.flatten(1).T

        if self.standardize:
            U_test = self.scaler.transform(U_test.T).T # TODO: why transform instead of fit_transform?

        X_test = fixpoint_iteration(self.weights['A'], self.weights['B'], U_test, self.device).cpu().numpy()
        Y_test_pred = self.weights['C'] @ X_test + self.weights['D'] @ U_test
        test_accuracy = np.mean(
            np.argmax(Y_test_pred, axis=0) == labels.numpy()
        )

        return test_accuracy

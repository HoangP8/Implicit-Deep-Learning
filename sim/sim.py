"""
Main SIM class that handles defining and training a SIM model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from contextlib import contextmanager
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Union

from solvers import solve
from utils import fixpoint_iteration

logger = logging.getLogger(__name__)

class HookManager:
    def __init__(self):
        self.pre_activations = []
        self.post_activations = []
        self.hooks = []

    def hook_fn(self, module, input, output):
        if input[0].size() == output.size():
            self.pre_activations.append(input[0].detach())
            self.post_activations.append(output.detach())

    def _apply_hooks(self, module, list_relu):
        children = list(module.children())
        if len(children) == 0:  # It's a leaf module
            if isinstance(module, (torch.nn.ReLU, torch.nn.GELU, torch.nn.Sigmoid)):
                list_relu.append(module)
        else:
            for child in module.children():
                self._apply_hooks(child, list_relu)  # Recursively apply to children

    def _apply_all_hooks(self, module):
        children_length = len(list(module.children()))
        if children_length == 0:
            if isinstance(module, (torch.nn.ReLU, torch.nn.GELU, torch.nn.Sigmoid)):
                hook = module.register_forward_hook(self.hook_fn)
                self.hooks.append(hook)
        elif children_length <= 3:
            for child in module.children():
                self._apply_all_hooks(child)
        else:
            for child in module.children():
                relu_layers = []
                self._apply_hooks(child, relu_layers)
                if relu_layers:
                    final_relu = relu_layers[-1]
                    hook = final_relu.register_forward_hook(self.hook_fn)
                    self.hooks.append(hook)

    # def _apply_all_hooks(self, module):
    #     children = list(module.children())
    #     if len(children) == 0:  # It's a leaf module
    #         if isinstance(module, torch.nn.ReLU) or isinstance(module, torch.nn.GELU):
    #             hook = module.register_forward_hook(self.hook_fn)
    #             self.hooks.append(hook)
    #     else:
    #         for child in module.children():
    #             self._apply_all_hooks(child)  # Recursively apply to children

    @contextmanager
    def register_hooks(self, model):
        self._apply_all_hooks(model)
        try:
            yield
        finally:
            for hook in self.hooks:
                hook.remove()
            self.pre_activations.clear()
            self.post_activations.clear()
            self.hooks.clear()


class SIM():
    """
    SIM base class.

    Args:
        config (dict): Configuration dictionary.
        device (str or torch.device, optional): Device to use for SIM. Defaults to None.
        dtype (str or torch.dtype, optional): Data type for SIM. Defaults to None.
        standardize (bool, optional): Whether to standardize the input data using scipy StandardScaler. Defaults to False.
    """
    def __init__(
        self, 
        config : dict,
        device : Optional[Union[str, torch.device]] = None, 
        dtype : Optional[Union[str, torch.dtype]] = None,
        standardize : bool = False,
    ):
        super(SIM, self).__init__()
        self.config = config
        self.device = device
        self.dtype = dtype
        self.weights = {
            'A': None,
            'B': None,
            'C': None,
            'D': None,
        }

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
    
    def get_states(
            self, 
            model, 
            dataloader : torch.utils.data.DataLoader,
    ) -> Dict[str, np.ndarray]:
        model.requires_grad_(False)
        model.eval()
        
        hooks = HookManager()

        outputs_accumulated = []
        inputs_accumulated = []
        pre_activations_accumulated = []
        post_activations_accumulated = []

        for input_samples, _ in dataloader:
            input_samples = input_samples.to(model.device)

            # Run the model with hooks and no gradient calculations
            with hooks.register_hooks(model), torch.no_grad():
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

        return states_data

    def forward(
        self, 
        input : torch.Tensor,
    ) -> torch.Tensor:

        for weight in self.weights.keys():
            assert self.weights[weight] is not None, f"Weight matrix {weight} is not trained"

        X = fixpoint_iteration(self.weights['A'], self.weights['B'], input, self.config.activation, self.device)
        Y = X @ self.weights['C'] + self.weights['D'] @ input
        return Y
    
    def train(
        self, 
        config : dict,
        states_data : Dict[str, np.ndarray],
    ):
        logger.info("===== Start training SIM =====")
        start_time = time.time()
        A, B, C, D = solve(states_data['X'], states_data['U'], states_data['Z'], states_data['Y'], config)
        end_time = time.time()
        logger.info(f"===== Training SIM finished in {(end_time - start_time):.4f} seconds =====")

        self.weights['A'] = A
        self.weights['B'] = B
        self.weights['C'] = C
        self.weights['D'] = D

        return self
    
    def eval(
        self,
        inputs : torch.Tensor,
        labels : torch.Tensor,
    ) -> float:
        """
        Evaluate the SIM model on the given test inputs and corresponding labels.

        Args:
            inputs (torch.Tensor): Input data with shape (batch_size, input_dim).
            labels (torch.Tensor): Labels for the input data with shape (batch_size,).
        
        Returns:
            test_accuracy (float): Accuracy of the SIM model on the given test inputs.
        """
        U_test = inputs.flatten(1).T

        X_test = fixpoint_iteration(self.weights['A'], self.weights['B'], U_test, self.device).cpu().numpy()
        Y_test_pred = self.weights['C'] @ X_test + self.weights['D'] @ U_test
        test_accuracy = np.mean(
            np.argmax(Y_test_pred, axis=0) == labels.numpy()
        )

        return test_accuracy

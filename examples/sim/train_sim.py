"""

This file contains the example code for training a simple Implicit Model with SIM.

"""

import torch
import random
from torch.utils.data import DataLoader, Subset

from idl.sim import SIM
from idl.sim.solvers import ADMMSolver

from explicit_networks import FashionMNIST_FFNN
from train_explicit import load_data


train_loader, test_loader = load_data(data_dir="data")

# Take only a subset of the training dataset to train the state-driven model
selected_indices = random.sample(
    range(len(train_loader.dataset)), 2000
)
subset = Subset(train_loader.dataset, selected_indices)
subset_loader = DataLoader(subset, batch_size=1000, shuffle=True)

explict_model = FashionMNIST_FFNN(28 * 28, 10)
explict_model.load_state_dict(torch.load("models/explicit_model.pt"))

sim = SIM(activation_fn=torch.nn.functional.relu, device="cuda", dtype=torch.float32)

solver = ADMMSolver(
    num_epoch_ab=1500,
    num_epoch_cd=120,
    lambda_y=1.0,
    lambda_z=1.0,
    rho_ab=1.0,
    rho_cd=1.0,
    batch_feature_size=120,
    regen_states=False,
)

# Train SIM
sim.train(solver=solver, model=explict_model, dataloader=subset_loader)

# Evaluate SIM
test_acc = sim.evaluate(test_loader)
print(f"Test accuracy: {test_acc}")

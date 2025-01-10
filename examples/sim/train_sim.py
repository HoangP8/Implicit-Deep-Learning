import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from idl.sim import SIM
from idl.sim.solvers import LeastSquareSolver
from .explicit_networks import FashionMNIST_FFNN

def load_data(data_dir="data"):
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.FashionMNIST(
        f"{data_dir}/FashionMNIST",
        train=True,
        download=True,
        transform=transform,
    )
    test_set = datasets.FashionMNIST(
        f"{data_dir}/FashionMNIST",
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=128,
        shuffle=False,
        num_workers=4,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=128,
        shuffle=False,
        num_workers=4,
    )
    return train_loader, test_loader

from idl.sim import SIM
from idl.sim.solvers import ADMMSolver

train_loader, test_loader = load_data()

explict_model = FashionMNIST_FFNN(28 * 28, 10)
explict_model.load_state_dict(torch.load("models/explicit_model.pth"))

sim = SIM(activation_fn=torch.nn.ReLU, device="cuda", dtype=torch.float32)

solver = ADMMSolver(
    num_epoch_ab=1500,
    num_epoch_cd=120,
    rho_ab=1.0,
    rho_cd=1.0,
    batch_feature_size=120,
    regen_states=False,
)
# Train SIM
sim.train(solver=solver, explict_model, train_loader.dataset.data)

# Evaluate SIM
sim.evaluate(test_loader.dataset.data, test_loader.dataset.targets)
from pathlib import Path
import torchvision
import torchvision.transforms as transforms
import os
import torch
from torch.utils.data import DataLoader


__file_path = os.path.abspath(__file__)
__proj_dir = "/".join(str.split(__file_path, "/")[:-2]) + "/"
DATA_PATH = Path(__proj_dir)


def cifar_load(train_bs, valid_bs=10000):
    PATH = DATA_PATH / "data" / "cifar"
    train_ds = torchvision.datasets.CIFAR10(root=PATH, train=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), transforms.Lambda(lambda x: torch.flatten(x))]), download=True)
    train_dl = DataLoader(train_ds, batch_size=train_bs, shuffle=True)
    valid_ds = torchvision.datasets.CIFAR10(root=PATH, train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), transforms.Lambda(lambda x: torch.flatten(x))]))
    valid_dl = DataLoader(valid_ds, batch_size=valid_bs, shuffle=True)

    return train_ds, train_dl, valid_ds, valid_dl


from os.path import dirname, abspath
import sys, os
d = dirname(dirname(abspath(__file__)))
sys.path.append(d)

from torch import optim
import torch.nn.functional as F
from implicit_model import ImplicitModel
from utils import train


epochs = 10
bs = 100
lr = 5e-4

train_ds, train_dl, valid_ds, valid_dl = cifar_load(bs)

n = 300 # the main parameter of an implicit model, determining the size of the hidden state matrix X 
p = 3072 # the flattened input size, in this case 32 x 32 (pixels) x 3 (channels) for CIFAR
q = 10 # the output size


# model = ImplicitModel(n, p, q)
model = ImplicitModel(n, p, q, mitr=1, grad_mitr=1, tol=1e-5, grad_tol=1e-7, activation='silu')
print(f"mitr: {model.f.mitr}, grad_mitr: {model.f.grad_mitr}, tol: {model.f.tol}, grad_tol: {model.f.grad_tol}, activation: {model.f.activation} ")
opt = optim.Adam(ImplicitModel.parameters(model), lr=lr)
loss_fn = F.cross_entropy
model, _ = train(model, train_dl, valid_dl, opt, loss_fn, epochs, "CIFAR_Implicit_300_Inf", device="cuda:0")
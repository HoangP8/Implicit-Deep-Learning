"""

This file contains a simple example code for training an explicit network, which is later used to generate states to train the Implicit Model.

"""

import os
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from explicit_networks import FashionMNIST_FFNN

logger = logging.getLogger(__name__)

def load_data(data_dir="data", batch_size=128, num_workers=4):
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
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, test_loader


def train(model, train_loader, loss_fn, optimizer, epoch, device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            loss = loss.item()
            logger.info(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss / len(data),
                )
            )

def test(model, test_loader, loss_fn, epoch, device):
    model.eval()
    test_loss = torch.tensor(0.0, device=device)
    correct = torch.tensor(0.0, device=device)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target)
            _, preds = torch.max(output, 1)
            correct += torch.sum(preds == target.data).item()

    test_loss = (test_loss / len(test_loader.dataset)).item()
    correct = correct.item()

    logger.info(
        "Test Epoch: {} Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            epoch,
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--output_dir", type=str, default="models")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.output_dir, exist_ok=True)

    model = FashionMNIST_FFNN(28 * 28, 10)
    
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Number of parameters: {num_params}")

    train_loader, test_loader = load_data(data_dir=args.data_dir, batch_size=args.batch_size)

    loss_fn = nn.CrossEntropyLoss(reduction="sum")

    train_params = [param for param in model.parameters()]
    logger.info(f"Number of trainable parameters: {sum(p.numel() for p in train_params)}")

    optimizer = optim.SGD(train_params, lr=args.learning_rate, momentum=0.9, weight_decay=0.0001)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.0001)

    for epoch in tqdm(range(args.epochs)):
        train(model, train_loader, loss_fn, optimizer, epoch, device)
        test(model, test_loader, loss_fn, epoch, device)
        scheduler.step()

    torch.save(model.state_dict(), f"{args.output_dir}/explicit_model.pt")

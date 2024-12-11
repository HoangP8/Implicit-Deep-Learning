from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

def load_data(args):
    """
    Load training and testing data loaders for the specified dataset.
    """
    
    # MNIST Dataset
    if args.dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),
            transforms.Lambda(lambda x: torch.flatten(x))
            ])
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # CIFAR-10 Dataset
    elif args.dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Lambda(lambda x: torch.flatten(x))  
        ])
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    return train_loader, test_loader

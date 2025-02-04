import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from .load_data import load_data
from .utils import train, set_seed
import os
from idl import ImplicitModel

def parse_args():
    parser = argparse.ArgumentParser(description="Train an Implicit Base model on MNIST or CIFAR-10")
    
    # Dataset
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10'], required=True, help="Dataset to use: 'mnist' or 'cifar10'")
    
    # Optimization parameters
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=100, help="Batch size for training and testing")
    parser.add_argument('--device', type=int, default=0, help="Specify the device id (e.g., 0 for cuda:0)")
    parser.add_argument('--lr', type=float, default=5e-3, help="Learning rate for the optimizer")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    
    # Implicit model parameters
    parser.add_argument('--hidden_dim', type=int, default=None, help="Hidden dimension of Implicit model")
    parser.add_argument('--mitr', type=int, default=300, help="Max iterations")
    parser.add_argument('--grad_mitr', type=int, default=300, help="Max gradient iterations")
    parser.add_argument('--tol', type=float, default=3e-6, help="Tolerance for convergence")
    parser.add_argument('--grad_tol', type=float, default=3e-6, help="Gradient tolerance for convergence")
    parser.add_argument('--is_low_rank', type=bool, default=False, help="Enable low-rank configuration for the model")
    parser.add_argument('--rank', type=int, default=1, help="The number of rank for low-rank configuration")
    parser.add_argument('--v', type=float, default=0.95, help="Inf ball")
    
    return parser.parse_args()

def main():
    """
    Main function to train the Implicit Base model.
    """

    args = parse_args()
    set_seed(args.seed)
    train_loader, test_loader = load_data(args)

    if args.hidden_dim is None:
        raise ValueError("Error: 'hidden_dim' must be specified for the model.")

    if args.dataset == 'mnist':
        input_dim = 784 
        output_dim = 10
    
    elif args.dataset == 'cifar10':
        input_dim = 3072 
        output_dim = 10 

    # Initialize the Implicit Base model
    model = ImplicitModel(
        hidden_dim=args.hidden_dim, 
        input_dim=input_dim, 
        output_dim=output_dim,
        mitr=args.mitr, 
        grad_mitr=args.grad_mitr, 
        tol=args.tol, 
        grad_tol=args.grad_tol,
        is_low_rank=args.is_low_rank, 
        rank=args.rank)

    # Define loss function, optimizer, device and log directory
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = F.cross_entropy
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    log_dir = os.path.join("results", f"im_{args.dataset}_{args.hidden_dim}")
    
    # Train the model
    model, log_file = train(
        args=args,
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        num_epochs=args.epochs,
        log_dir=log_dir,
        device=device
    )
    print(f"Training complete. Logs saved to {log_file}")

if __name__ == '__main__':
    main()

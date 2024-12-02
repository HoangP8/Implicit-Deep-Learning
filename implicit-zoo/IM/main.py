import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from .load_data import load_data
from .utils import train
import os
import sys
sys.path.append('../implicit')
from implicit.implicit_model import ImplicitModel

def parse_args():
    parser = argparse.ArgumentParser(description="Train an implicit model on MNIST or CIFAR-10")
    
    # Add arguments for dataset, model, training parameters, etc.
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10'], required=True, help="Dataset to use: 'mnist' or 'cifar10'")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=100, help="Batch size for training and testing")
    parser.add_argument('--hidden_size', type=int, default=None, help="Hidden size of Implicit model")
    parser.add_argument('--lr', type=float, default=5e-3, help="Learning rate for the optimizer")
    parser.add_argument('--device', type=int, default=0, help="Specify the device id (e.g., 0 for cuda:0)")
    parser.add_argument('--mitr', type=int, default=300, help="Max iterations")
    parser.add_argument('--grad_mitr', type=int, default=300, help="Max gradient iterations")
    parser.add_argument('--tol', type=float, default=3e-6, help="Tolerance for convergence")
    parser.add_argument('--grad_tol', type=float, default=3e-6, help="Gradient tolerance for convergence")
    parser.add_argument('--activation', type=str, default='relu', help="Activation for implicit layers")
    
    return parser.parse_args()

def main():
    args = parse_args()
    train_loader, test_loader = load_data(args)

    if args.hidden_size is None:
        raise ValueError("Error: 'hidden_size' must be specified for the model.")

    if args.dataset == 'mnist':
        input_size = 784 
        output_size = 10
    
    elif args.dataset == 'cifar10':
        input_size = 3072 
        output_size = 10 

    model = ImplicitModel(hidden_size=args.hidden_size, input_size=input_size, output_size=output_size,
                          mitr=args.mitr, grad_mitr=args.grad_mitr, tol=args.tol, grad_tol=args.grad_tol,
                          activation=args.activation)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = F.cross_entropy
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    log_dir = os.path.join("results", f"im_{args.dataset}_{args.hidden_size}")
    
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

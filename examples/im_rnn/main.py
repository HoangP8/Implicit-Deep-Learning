import argparse
import torch
from .load_data import load_data
import sys
from .utils import train_time_series
import os

sys.path.append('../implicit')
from idl.implicit_rnn_model import ImplicitRNN

def parse_args():
    parser = argparse.ArgumentParser(description="Train an implicit model on MNIST or CIFAR-10")
    
    # Dataset and training parameters
    parser.add_argument('--dataset', type=str, choices=['netflix', 'spiky'], required=True, help="Dataset to use: 'netflix' or 'spiky'")
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs for training")
    parser.add_argument('--look_back', type=int, default=60, help="Look-back period for time series data")
    
    # Optimization parameters
    parser.add_argument('--lr', type=float, default=5e-3, help="Learning rate for the optimizer")
    parser.add_argument('--device', type=int, default=0, help="Specify the device id (e.g., 0 for cuda:0)")
    
    # Implicit model parameters
    parser.add_argument('--hidden_dim', type=int, default=None, help="Hidden size of RNN model")
    parser.add_argument('--implicit_hidden_dim', type=int, default=None, help="Hidden size of Implicit layer")    
    parser.add_argument('--input_dim', type=int, default=1, help="Input size")
    parser.add_argument('--output_dim', type=int, default=1, help="Output size")
    parser.add_argument('--is_low_rank', type=bool, default=False, help="Whether to use low rank approximation (True/False)")
    parser.add_argument('--rank', type=int, default=1, help="Rank for low rank approximation (used only if is_low_rank is True)")
    parser.add_argument('--mitr', type=int, default=500, help="Max iterations")
    parser.add_argument('--grad_mitr', type=int, default=500, help="Max gradient iterations")
    parser.add_argument('--tol', type=float, default=1e-6, help="Tolerance for convergence")
    parser.add_argument('--grad_tol', type=float, default=1e-6, help="Gradient tolerance for convergence")
    parser.add_argument('--v', type=float, default=0.95, help="Inf ball")
    
    return parser.parse_args()

def main():
    """
    Main function to train the Implicit RNN model.

    This function performs the following steps:
        1. Parses command-line arguments.
        2. Loads the specified dataset.
        3. Initializes the Implicit RNN model with the provided parameters.
        4. Sets up the optimizer and loss function.
        5. Trains the model and logs the training progress.
        6. Outputs the model size and training completion message.
    """

    args = parse_args()
    x_train, x_test, y_train, y_test = load_data(args)

    if args.hidden_dim is None:
        raise ValueError("Error: 'hidden_size' must be specified for the model.")
    elif args.implicit_hidden_dim is None:
        raise ValueError("Error: 'implicit_hidden_dim' must be specified for the model.")
    
    # Initialize the Implicit RNN model
    model = ImplicitRNN(
        hidden_dim=args.hidden_dim,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        is_low_rank=args.is_low_rank,
        rank=args.rank,
        implicit_hidden_dim=args.implicit_hidden_dim,
        mitr=args.mitr,
        grad_mitr=args.grad_mitr,
        tol=args.tol,
        grad_tol=args.grad_tol,
        v=args.v
    )
    
    print(f'Model size: {sum(p.numel() for p in model.parameters())} parameters')
    
    # Define loss function, optimizer, device and log directory
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu") 
    log_dir = os.path.join("results", f"im_rnn_{args.dataset}")
    
    # Train the model
    train_time_series(args, model, x_train, y_train, x_test, y_test, optimizer, loss_fn, log_dir, device)

if __name__ == '__main__':
    main()

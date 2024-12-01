import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from .load_data import load_data
from .utils import train
import sys
sys.path.append('../implicit')
from implicit.implicit_model import ImplicitModel

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model on MNIST or CIFAR-10")
    
    # Add arguments for dataset, model, training parameters, etc.
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10'], required=True, help="Dataset to use: 'mnist' or 'cifar10'")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=100, help="Batch size for training and testing")
    parser.add_argument('--lr', type=float, default=5e-3, help="Learning rate for the optimizer")
    parser.add_argument('--log_dir', type=str, default='./logs', help="Directory to save log files")
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda', help="Device for training (cpu or cuda)")

    return parser.parse_args()

def main():
    args = parse_args()
    train_loader, test_loader = load_data(args)

    if args.dataset == 'mnist':
        n = 100 
        p = 784 
        q = 10 
    elif args.dataset == 'cifar10':
        n = 300 
        p = 3072 
        q = 10 

    model = ImplicitModel(hidden_size=n, input_size=p, output_size=q)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = F.cross_entropy

    model, log_file = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        num_epochs=args.epochs,
        log_dir=args.log_dir,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )

    print(f"Training complete. Logs saved to {log_file}")

if __name__ == '__main__':
    main()

from .netflix_stock import netflix_dataset
from .spiky_data import spiky_synthetic_dataset

def load_data(args):
    """
    Load training and testing data loaders for the specified dataset.
    """

    # Netflix Dataset
    if args.dataset == 'netflix':
        x_train, x_test, y_train, y_test = netflix_dataset(args.look_back)

    # Spiky Synthetic Dataset
    elif args.dataset == 'spiky':
        x_train, x_test, y_train, y_test = spiky_synthetic_dataset(args.look_back)
    
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    return x_train, x_test, y_train, y_test
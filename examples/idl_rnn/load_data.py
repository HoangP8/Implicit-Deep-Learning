from .spiky_data import spiky_synthetic_dataset

def load_data(args):
    """
    Load training and testing data loaders for the specified dataset.
    """

    # Spiky Synthetic Dataset
    if args.dataset == 'spiky':
        x_train, x_test, y_train, y_test = spiky_synthetic_dataset(args.look_back)
    
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    return x_train, x_test, y_train, y_test
import torch
import datetime
import os

def transpose(X):
    """
    Transpose a 2D matrix.
    """
    
    assert len(X.size()) == 2, "data must be 2D"
    return X.T


def get_test_accuracy(model, loss_fn, test_loader, device):
    """
    Evaluate the model on the test dataset and compute loss and accuracy.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        loss_fn (torch.nn.Module): The loss function used for evaluation.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to perform computation on (CPU or GPU).

    Returns:
        Tuple[float, float]: Average loss and accuracy on the test dataset.
    """
    
    model.eval() # eval mode
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():  # Disable gradient 
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs)
            if not isinstance(predictions, torch.Tensor):
                predictions = torch.from_numpy(predictions).to(device)

            # Compute loss
            loss = loss_fn(predictions, targets)
            total_loss += loss.item()

            # Compute accuracy
            predicted_labels = torch.argmax(predictions, dim=-1)
            correct_predictions += (predicted_labels == targets).sum().item()
            total_samples += targets.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = correct_predictions / total_samples
    model.train() # train mode
    return avg_loss, accuracy


def train(args, model, train_loader, test_loader, optimizer, loss_fn, log_dir, device):
    """
    Trains the model and logs the results.

    Args:
        args: Configuration object with training parameters.
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        test_loader (DataLoader): DataLoader for the test dataset.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        loss_fn (torch.nn.Module): Loss function to minimize.
        log_dir (str): Directory to save log files.
        device (torch.device): Device to perform training on (CPU or GPU).
        
    Returns:
        Tuple[torch.nn.Module, str]: The trained model and the path to the log file.
    """

    model.to(device)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f"{timestamp}.log")
    with open(log_file, 'w') as f:
        f.write("Arguments:\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
        f.write("\n")
        f.write(f'Model size: {sum(p.numel() for p in model.parameters())} parameters\n')
        f.write("\n")
        f.write("Epoch,Train Loss,Train Accuracy,Test Loss,Test Accuracy\n")

    # Training loop
    for epoch in range(args.epochs):
        model.train()  # train mode
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        batch_idx = 0
        total_batches = len(train_loader) 

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Compute training accuracy
            predicted_labels = torch.argmax(predictions, dim=-1)
            correct_train += (predicted_labels == targets).sum().item()
            total_train += targets.size(0)
            train_loss += loss.item()

            # Log every 100 batches
            if batch_idx % 100 == 0:
                test_loss, test_accuracy = get_test_accuracy(model, loss_fn, test_loader, device)

                print(f"Epoch {epoch + 1}/{args.epochs} - "
                      f"Batch {batch_idx + 1}/{total_batches}: "
                      f"Train Loss: {loss.item():.4f}, Train Accuracy: {100 * correct_train / total_train:.2f}%, "
                      f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy * 100:.2f}%")

                with open(log_file, 'a') as f:
                    f.write(f"Epoch {epoch + 1}/{args.epochs} - "
                            f"Batch {batch_idx + 1}/{total_batches}: "
                            f"Train Loss: {loss.item():.4f}, Train Accuracy: {100 * correct_train / total_train:.2f}%, "
                            f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy * 100:.2f}%\n")

            batch_idx += 1

        # Compute average training metrics for each epoch
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = (correct_train / total_train) * 100
        test_loss, test_accuracy = get_test_accuracy(model, loss_fn, test_loader, device)

        # Log epoch results
        print(f"Epoch {epoch + 1}/{args.epochs}: "
              f"Train Loss: {avg_train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy * 100:.2f}%")
        with open(log_file, 'a') as f:
            f.write(f"Epoch {epoch + 1}/{args.epochs}: "
                    f"Train Loss: {avg_train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}% | "
                    f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy * 100:.2f}%\n")

        with open(log_file, 'a') as f:
            f.write("-" * 100 + "\n")
        print("-" * 100)

    return model, log_file
import math
import time
import torch
import torch.nn as nn
import os
import logging
import random
import torch
import numpy as np


def set_seed(seed):
    """
    Set seed for reproducibility.
    """
    random.seed(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def print_model_size(model):
    """
    Print the total number of trainable parameters in a model.
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params}")
    

def get_batch(data, block_size, batch_size, device):
    """
    Create a batch of data for training or evaluation.
    
    Args:
    - data: Tensor containing sequence data.
    - block_size: Number of data points in each sequence.
    - batch_size: Number of sequences per batch.
    - device: Device to load the batch.
    
    Returns:
    - Input and target batch tensors.
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, data, block_size, batch_size, device, eval_iters):
    """
    Estimate the average loss of a model over specified iterations.
    
    Args:
    - model: Model to evaluate.
    - data: Data dictionary with 'train' and 'val' data.
    - block_size: Sequence length.
    - batch_size: Number of sequences in each batch.
    - device: Computation device.
    - eval_iters: Number of iterations to average over.
    
    Returns:
    - Average loss dictionary for training and validation data.
    """
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data[split], block_size, batch_size, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train_model(args, model, data, device, model_type="", log_file=None, write_initial=True):
    """
    Train the given model on the provided dataset.

    Args:
        model: The model to train.
        data: Dictionary containing training and validation datasets.
        args: Configurations from main.py
        device: Device used to train.
        model_type: Choose type of model to train: Explicit or Implicit, default is "".
        log_file: Path to the log file.
        write_initial: If True, writes the initial arguments and model size.
    """

    if write_initial:
        with open(log_file, 'w') as f:
            f.write("Arguments:\n")
            for arg, value in vars(args).items():
                f.write(f"{arg}: {value}\n")
            f.write("\n")
            f.write(f'Model size: {sum(p.numel() for p in model.parameters())} parameters\n')
            f.write("\n")

    checkpoints_dir = './checkpoints'
    os.makedirs(checkpoints_dir, exist_ok=True)

    start_time = time.time()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    print(f"Training {model_type} GPT model")
    logging.info(f"Training {model_type} GPT model")
    with open(log_file, 'a') as f:
        f.write(f"Training {model_type} GPT model\n")

    for iter in range(args.max_iters):
        if iter % args.eval_interval == 0 or iter == args.max_iters - 1:
            losses = estimate_loss(model, data, args.block_size, args.batch_size, device, args.eval_iters)
            log_message = (
                f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f},"
                f" train perplexity {math.exp(losses['train']):.4f}, val perplexity {math.exp(losses['val']):.4f}"
            )
            print(log_message)
            with open(log_file, 'a') as f:
                f.write(f"{log_message}\n")
        
        xb, yb = get_batch(data['train'], args.block_size, args.batch_size, device)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    end_time = time.time()
    training_time = end_time - start_time
    log_time_message = f"Training time for {model_type} model (seconds): {training_time:.2f}"
    print(log_time_message)
    with open(log_file, 'a') as f:
        f.write(f"{log_time_message}\n")
    
    model_filename = f"gpt_char_{args.dataset}_{model_type.lower()}_iter_{args.max_iters}.pt"
    model_path = os.path.join(checkpoints_dir, model_filename)

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    with open(log_file, 'a') as f:
        f.write(f"Model saved to {model_path}\n")
        f.write("\n")


def generate_text(args, idl_model, additional_data, device):
    """
    Generate text using a pre-trained language model.
    
    Args:
    - args: Configurations from main.py
    - idl_model: Trained implicit language model.
    - additional_data: Data needed for generation (e.g., tokenizer).
    - device: Device for running the model.
    
    Returns:
    - The text generated by the model.
    """
    if args.dataset == "tinyshakespeare":
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        generated_ids = idl_model.generate(context, args.max_new_tokens, args.block_size)[0].tolist()
        generated_text_idl = "".join([additional_data[i] for i in generated_ids])

    elif args.dataset == "tinystories":
        context = torch.tensor(additional_data.encode('\n'), dtype=torch.long, device=device).unsqueeze(0)
        generated_text_idl = additional_data.decode(
            idl_model.generate(context, args.max_new_tokens, args.block_size)[0].tolist()
        )

    elif args.dataset == "wikitext":
        context = torch.tensor(additional_data.encode_ordinary("\n"), dtype=torch.int, device=device).unsqueeze(0)
        generated_text_idl = additional_data.decode(
            idl_model.generate(context, args.max_new_tokens, args.block_size)[0].tolist()
        )

    else:
        raise NotImplementedError(f"Text generation for dataset {args.dataset} is not supported.")

    return generated_text_idl
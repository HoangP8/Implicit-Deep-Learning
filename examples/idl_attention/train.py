import math
import time
import torch
from torch import nn, Tensor
import os
from typing import Any, Dict, Optional
from .utils import estimate_loss, get_batch
from idl import IDLHead


def initialize_idl_heads(args: Any, idl_model: nn.Module) -> None:
    """
    Initialize IDL attention heads in the model.

    Args:
        args (Any): Configurations from main.py
        idl_model (nn.Module): The model to initialize with IDL heads.
    """
    
    for i in range(args.n_layer):
        idl_model.blocks[i].sa.heads = nn.ModuleList(
            [
                IDLHead(
                    args.n_embd // args.n_head,
                    args.n_embd,
                    args.block_size,
                    args.fixed_point_iter,
                    args.attention_version,
                    args.is_low_rank,
                    args.rank
                )
                for _ in range(args.n_head)
            ]
        )


def train_model(
    args: Any,
    model: nn.Module,
    data: Dict[str, Tensor],
    device: torch.device,
    log_file: Optional[str] = None
) -> None:
    """
    Train the given model on the provided dataset.

    Args:
        args (Any): Configurations from main.py containing attributes such as 'max_iters',
                    'lr', 'block_size', 'batch_size', 'eval_interval', 'eval_iters',
                    and 'dataset'.
        model (nn.Module): The model to train.
        data (Dict[str, Tensor]): Dictionary containing training and validation datasets with keys 'train' and 'val'.
        device (torch.device): Device used to train (CPU or GPU).
        log_file (Optional[str]): Path to the log file. If None, logs will not be saved to a file.
    """

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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"Training Implicit GPT model")
    with open(log_file, 'a') as f:
        f.write(f"Training Implicit GPT model\n")

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
    log_time_message = f"Training time for Implicit model (seconds): {training_time:.2f}"
    print(log_time_message)
    with open(log_file, 'a') as f:
        f.write(f"{log_time_message}\n")
    
    model_filename = f"gpt_char_{args.dataset}_implicit_iter_{args.max_iters}.pt"
    model_path = os.path.join(checkpoints_dir, model_filename)

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    with open(log_file, 'a') as f:
        f.write(f"Model saved to {model_path}\n")
        f.write("\n")
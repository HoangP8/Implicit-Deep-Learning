import math
import time
import torch
import torch.nn as nn
import os
from .utils import estimate_loss, get_batch
from .model import IDLHead
import logging


def copy_weights(src_head, tgt_head, head_size, enforce_structure_IDL, n_embd):
    """
    Copy the weights from the source model to the target model.

    Args:
        src_head: Source attention head.
        tgt_head: Target attention head.
        head_size: Size of a single attention head.
        enforce_structure_IDL: Boolean if the IDL structure should be enforced.
        n_embd: Embedding dimension
    """
    if enforce_structure_IDL is True:
        tgt_head.A.data.fill_(0)
        tgt_head.A.data[:, 3 * head_size :] = 1

        Z = torch.zeros(
            (n_embd, head_size), device=src_head.key.weight.device, requires_grad=False
        )
        tgt_head.B.data.copy_(
            torch.cat(
                [
                    src_head.key.weight.data.T,
                    src_head.query.weight.data.T,
                    src_head.value.weight.data.T,
                    Z,
                ],
                dim=1,
            )
        )

        tgt_head.A.requires_grad = False
        tgt_head.B[:, 3 * head_size :].detach()
    else:
        tgt_head.B.data.copy_(
            torch.cat(
                [
                    src_head.key.weight.data.T,
                    src_head.query.weight.data.T,
                    src_head.value.weight.data.T,
                    tgt_head.B[:, 3 * head_size :],
                ],
                dim=1,
            )
        )
        

def initialize_idl_heads(args, idl_model):
    """
    Initialize IDL attention heads in the model.

    Args:
        args: Configurations from main.py
        idl_model: The model to initialize with IDL heads.
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
        
def copy_non_attention_parameters(source_model, target_model):
    """"
    Copy non-attention parameters from the source model to the target model.

    Args:
        source_model: The source model containing the parameters to copy.
        target_model: The target model to receive the parameters.
    """

    target_model.token_embedding_table.load_state_dict(source_model.token_embedding_table.state_dict())
    target_model.position_embedding_table.load_state_dict(source_model.position_embedding_table.state_dict())
    target_model.lm_head.load_state_dict(source_model.lm_head.state_dict())
    target_model.ln_f.load_state_dict(source_model.ln_f.state_dict())
    
    for i in range(len(source_model.blocks)):
        target_model.blocks[i].ln1.load_state_dict(source_model.blocks[i].ln1.state_dict())
        target_model.blocks[i].ln2.load_state_dict(source_model.blocks[i].ln2.state_dict())
        target_model.blocks[i].ffwd.load_state_dict(source_model.blocks[i].ffwd.state_dict())
        target_model.blocks[i].sa.proj.load_state_dict(source_model.blocks[i].sa.proj.state_dict())
        target_model.blocks[i].sa.dropout.load_state_dict(source_model.blocks[i].sa.dropout.state_dict())


def load_or_train_explicit_model(args, model, data, device, log_file):
    """Loads or trains the explicit model."""
    
    if args.explicit_model_path:
        print("Loading an explicit GPT model")
        model.load_state_dict(torch.load(args.explicit_model_path))
    else:
        train_model(args, model, data, device, "Explicit", log_file)


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


def train_idl_model(args, model, idl_model, data, device, log_file):
    """
    Copy the attention parameters from explicit model to IDL model and train the IDL model.

    Args:
        model: Explicit GPT model.
        idl_model: Implicit GPT model with IDL heads.
        data: Dictionary containing training and validation datasets.
        args: Configurations from main.py
        device: Device usede to train.
        log_file: Path to the log file.
    """    
    
    # Copy attention parameters from explicit model
    for layer in range(args.n_layer):
        for head in range(args.n_head):
            copy_weights(
                model.blocks[layer].sa.heads[head],
                idl_model.blocks[layer].sa.heads[head],
                args.n_embd // args.n_head,
                args.enforce_structure_IDL,
                args.n_embd,
            )
    
    train_model(args, idl_model, data, device, "Implicit", log_file=log_file, write_initial=False)
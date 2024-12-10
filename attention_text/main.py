import argparse
import torch
from utils.data import load_data
from utils.model import GPTLanguageModel
from utils.utils import set_seed, generate_text
from utils.train import initialize_idl_heads, copy_non_attention_parameters, train_idl_model, load_or_train_explicit_model

#TODO
'''
1. Log the training effectively
2. Adjust the attention_version in the model code effectively
3. Debug for both explicit and implicit model
'''

def parse_args():
    parser = argparse.ArgumentParser(description="Train, evaluate and generate text with GPT implicit models.")
    
    # device id and seed
    parser.add_argument("--device", type=int, default=0, help="Specify the device id (e.g., 0 for cuda:0)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    
    # dataset and training configs
    parser.add_argument("--dataset", type=str, required=True, help="Dataset")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--block_size", type=int, default=256, help="Size of data blocks")
    parser.add_argument("--max_iters", type=int, default=30000, help="Max training iterations")
    parser.add_argument("--eval_interval", type=int, default=500, help="Interval for evaluation")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--eval_iters", type=int, default=200, help="Number of iterations for evaluation")
    parser.add_argument("--n_embd", type=int, default=384, help="Embedding size")
    parser.add_argument("--n_head", type=int, default=2, help="Number of attention heads")
    parser.add_argument("--n_layer", type=int, default=2, help="Number of layers")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    
    # IDL configs
    parser.add_argument("--fixed_point_iter", type=int, default=2, help="Fixed point iteration count")
    parser.add_argument("--enforce_structure_IDL", action="store_true", help="Enforce structured IDL model")
    parser.add_argument("--init_implicit_from_explicit", action="store_true", help="Initialize IDL model from explicit GPT")
    parser.add_argument("--explicit_model_path", type=str, help="Path to pre-trained explicit model")
    parser.add_argument("--attention_version", type=str, default="softmax", help="Attention mechanism version")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Max new tokens for generation")
    parser.add_argument("--is_low_rank", action="store_true", help="Low-rank approach for implicit model")
    parser.add_argument("--rank", type=int, default=1, help="Rank k of the low-rank approach")
    
    return parser.parse_args()


def main():
    """
    Main function to initialize models, train, evaluate and generate text.
    """

    args = parse_args()
    set_seed(args.seed)
    device = f"cuda:{args.device}"

    # Load data
    train_data, val_data, vocab_size, additional_data = load_data(args)
    data = {"train": train_data, "val": val_data}

    # Initialize models
    model = GPTLanguageModel(
        vocab_size, 
        args.n_embd, 
        args.block_size, 
        args.n_layer, 
        args.n_head, 
        args.dropout,
        args.attention_version
    ).to(device)

    idl_model = GPTLanguageModel(
        vocab_size, 
        args.n_embd, 
        args.block_size, 
        args.n_layer, 
        args.n_head, 
        args.dropout,
        args.attention_version
    )
    
    # Train/Load explicit model if needed and copy non-attention parameters from explicit to implicit model
    if args.init_implicit_from_explicit:
        load_or_train_explicit_model(args, model, data, device)
        copy_non_attention_parameters(idl_model, model)
        
    # Train IDL model
    initialize_idl_heads(args, idl_model)
    idl_model.to(device)
    train_idl_model(args, model, idl_model, data, device)
    print("Generated text with the trained IDL model:", generate_text(args, idl_model, additional_data, device))
    
    
if __name__ == "__main__":
    main()
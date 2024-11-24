import argparse
import torch
from utils.data import load_data
from utils.model import GPTLanguageModel
from utils.utils import set_seed, generate_text
from utils.train import train_model, initialize_idl_heads, copy_non_attention_parameters, copy_and_train_idl_model

def main(args):
    """
    Main function to initialize models, train, evaluate and generate text.
    """
    
    set_seed(args.seed)
    device = f"cuda:{args.device}"

    # Load data
    train_data, val_data, vocab_size, additional_data = load_data(args)
    data = {"train": train_data, "val": val_data}

    # Initialize models
    model = GPTLanguageModel(vocab_size, args.n_embd, args.block_size, args.n_layer, args.n_head, args.dropout).to(device)
    idl_model = GPTLanguageModel(vocab_size, args.n_embd, args.block_size, args.n_layer, args.n_head, args.dropout)

    # Train explicit model if needed and initialize IDL model
    if args.init_implicit_from_explicit:
        if args.explicit_model_path:
            print("Loading an explicit GPT model")
            model.load_state_dict(torch.load(args.explicit_model_path))
        else:
            train_model(model, data, args, device, "Explicit")

        # Copy non-attention parameters to IDL model
        copy_non_attention_parameters(idl_model, model)

    # Train IDL model
    initialize_idl_heads(args, idl_model)
    idl_model.to(device)
    copy_and_train_idl_model(model, idl_model, data, args, device)

    # Generate and display text
    generated_text = generate_text(args, idl_model, additional_data, device)
    print("Generated text with the trained IDL model:")
    print(generated_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train, evaluate and generate text with GPT implicit models.")
    parser.add_argument("--device", type=str, required=True, help="CUDA device index")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
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
    parser.add_argument("--fixed_point_iter", type=int, default=2, help="Fixed point iteration count")
    parser.add_argument("--enforce_structure_IDL", action="store_true", help="Enforce structured IDL model")
    parser.add_argument("--init_implicit_from_explicit", action="store_true", help="Initialize IDL model from explicit GPT")
    parser.add_argument("--explicit_model_path", type=str, help="Path to pre-trained explicit model")
    parser.add_argument("--attention_version", type=str, default="softmax", help="Attention mechanism version")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Max new tokens for generation")
    parser.add_argument("--is_low_rank", action="store_true", help="Low-rank approach for implicit model")
    parser.add_argument("--low_rank_k", type=int, default=1, help="Rank k of the low-rank approach")

    args = parser.parse_args()
    main(args)
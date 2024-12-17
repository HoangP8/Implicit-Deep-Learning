import torch
import torch.nn as nn
from typing import Tuple, Callable, Optional
from .utils import copy_non_attention_parameters, initialize_idl_heads, copy_weights

def ImplicitAttention(
    model_class: type,
    model_config: dict,
    device: torch.device,
    init_from_explicit: bool = False,
    explicit_ckpt: Optional[str] = None,
    train_model: Optional[Callable[[nn.Module], None]] = None,
    fixed_point_iter: Optional[int] = None,
    enforce_structure_IDL: bool = False,
    attention_version: str = 'softmax',
    is_low_rank: bool = False,
    rank: int = 1
) -> Tuple[nn.Module, nn.Module]:
    """
    Initializes explicit and implicit models. Optionally trains/loads the explicit model,
    copies non-attention parameters to the implicit model, and initializes IDL attention heads.

    Args:
        model_class (type): The model class to instantiate (e.g., GPTLanguageModel).
        model_config (dict): Configuration dictionary for model initialization. Expected to include:
            - 'vocab_size' (int): Vocabulary size.
            - 'n_embd' (int): Embedding dimension.
            - 'block_size' (int): Size of the input block.
            - 'n_layer' (int): Number of layers in the model.
            - 'n_head' (int): Number of attention heads per layer.
            - 'dropout' (float): Dropout rate.
            - 'attention_version' (str): Version of the attention mechanism.
        device (torch.device): Device to move the models to.
        init_from_explicit (bool, optional): Flag to initialize implicit model from explicit.
            Defaults to False.
        explicit_ckpt (str, optional): Path to the checkpoint to load the explicit model.
            If provided, the explicit model will be loaded from this checkpoint.
            Defaults to None.
        train_model (Callable[[nn.Module], None], optional): Function to train the explicit model.
            This function should accept the explicit model as its only argument.
            Defaults to None.
        fixed_point_iter (int, optional): Number of fixed point iterations.
            Required for IDL heads initialization.
            Defaults to None.
        enforce_structure_IDL (bool, optional): Whether to enforce the IDL structure during weight copying.
            Defaults to False.
        is_low_rank (bool, optional): Whether to use low-rank approximation in IDL heads.
            Defaults to False.
        rank (int, optional): Rank parameter for IDL heads.
            Defaults to 1.

    Returns:
        Tuple[nn.Module, nn.Module]: A tuple containing the explicit model and the implicit model.

    Raises:
        ValueError: 
            - If `init_from_explicit` is True and neither `explicit_ckpt` nor `train_model` is provided.
            - If required parameters for IDL heads initialization (`fixed_point_iter`) are missing.
        TypeError: If `train_model` is provided but is not callable.
    """
    # Initialize both explicit and implicit models
    explicit_model, implicit_model = [model_class(**model_config, attention_version=attention_version) for _ in range(2)]
    explicit_model.to(device)
    
    # If initialization from explicit is required
    if init_from_explicit:
        if train_model is None and explicit_ckpt is None:
            raise ValueError(
                "Either `train_model` or `explicit_ckpt` must be provided when `init_from_explicit` is True."
            )
        if train_model is not None and not callable(train_model):
            raise TypeError("`train_model` must be a callable function.")
        
        if explicit_ckpt:
            explicit_model.load_state_dict(torch.load(explicit_ckpt, map_location=device))
        else:
            train_model(explicit_model)
        
        copy_non_attention_parameters(implicit_model, explicit_model)

    # Check if all required parameters for IDL heads are provided
    required_config_keys = ['n_layer', 'n_head', 'n_embd', 'block_size']
    missing_keys = [key for key in required_config_keys if key not in model_config]
    if missing_keys:
        raise ValueError(f"Missing required model_config keys: {', '.join(missing_keys)}")
    
    n_layer = model_config['n_layer']
    n_head = model_config['n_head']
    n_embd = model_config['n_embd']
    block_size = model_config['block_size']

    if fixed_point_iter is None:
        raise ValueError("`fixed_point_iter` must be provided for IDL heads initialization.")

    # Initialize IDL attention heads
    initialize_idl_heads(
        idl_model=implicit_model,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        fixed_point_iter=fixed_point_iter,
        attention_version=attention_version,
        is_low_rank=is_low_rank,
        rank=rank
    )
    implicit_model.to(device)

    # copy weights
    for layer in range(n_layer):
        for head in range(n_head):
            copy_weights(
                explicit_model.blocks[layer].sa.heads[head],
                implicit_model.blocks[layer].sa.heads[head],
                n_embd // n_head,
                enforce_structure_IDL,
                n_embd,
            )

    return implicit_model
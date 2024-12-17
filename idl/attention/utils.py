import torch
import torch.nn as nn
from .IDLHead import IDLHead


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
        

def initialize_idl_heads(
    idl_model: nn.Module,
    n_layer: int,
    n_head: int,
    n_embd: int,
    block_size: int,
    fixed_point_iter: int,
    attention_version: str,
    is_low_rank: bool = False,
    rank: int = 1
) -> None:
    """
    Initialize IDL (Implicit Differentiable Learning) attention heads in the model.

    This function replaces the standard attention heads in each layer of the model's blocks
    with custom IDLHead modules, configured based on the provided parameters.

    Args:
        idl_model (nn.Module): The model to initialize with IDL attention heads.
        n_layer (int): Number of layers in the model.
        n_head (int): Number of attention heads per layer.
        n_embd (int): Embedding dimension.
        block_size (int): Size of the input block.
        fixed_point_iter (int): Number of fixed point iterations for IDL computation.
        attention_version (str): Version of the attention mechanism to use.
        is_low_rank (bool, optional): Whether to use low-rank approximation in IDL heads.
            Defaults to False.
        rank (int, optional): Rank parameter for low-rank approximation in IDL heads.
            Defaults to 1.
    """
    for layer_idx in range(n_layer):
        idl_model.blocks[layer_idx].sa.heads = nn.ModuleList([
            IDLHead(
                head_size=n_embd // n_head,
                n_embd=n_embd,
                block_size=block_size,
                fixed_point_iter=fixed_point_iter,
                attention_version=attention_version,
                is_low_rank=is_low_rank,
                rank=rank
            )
            for _ in range(n_head)
        ])


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
        
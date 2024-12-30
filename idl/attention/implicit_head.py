import torch
from torch import nn, Tensor
import torch.nn.functional as F


class IDLHead(nn.Module):
    """One head of self-attention implemented in IDL"""

    def __init__(
        self,
        head_size: int,
        n_embd: int,
        block_size: int,
        fixed_point_iter: int,
        attention_version: str,
        is_low_rank: bool = False,
        rank: int = 1
    ) -> None:
        """
        Initialize the IDL self-attention head.

        Args:
            head_size (int): The size of the attention head.
            n_embd (int): Embedding dimension.
            block_size (int): The block size for masking.
            fixed_point_iter (int): The number of iterations for fixed-point optimization in DEQ.
            attention_version (str): Type of attention mechanism to use. Choose 'softmax' or 'lipschitz'.
            is_low_rank (bool, optional): Whether to use a low-rank approach. Default is False.
            rank (int, optional): The rank for the low-rank approach. Default is 1.
        """

        super().__init__()
        self.hs = head_size
        self.is_low_rank = is_low_rank
        self.attention_version = attention_version
        
        if self.is_low_rank:
            self.rank = rank
            self.L = nn.Parameter(torch.zeros(self.hs * 4, self.rank))
            self.R = nn.Parameter(torch.zeros(self.rank, self.hs * 4))
            nn.init.xavier_uniform_(self.L)
            nn.init.xavier_uniform_(self.R)
            
        else:
            self.A = nn.Parameter(torch.zeros(self.hs * 4, self.hs * 4))  
            nn.init.xavier_uniform_(self.A)

        self.B = nn.Parameter(torch.zeros((n_embd, self.hs * 4)))
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))        

        nn.init.xavier_uniform_(self.B)
        self.fixed_point_iter = fixed_point_iter
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the IDL self-attention head.

        Args:
            x (Tensor): Input tensor of shape (B, T, C).

        Returns:
            Tensor: Output tensor after self-attention, of shape (B, T, hs).
        """

        B, T, C = x.shape
        B_U = torch.einsum("dk,bnd->bnk", self.B, x)
        X = torch.zeros((B, T, 4 * self.hs), device=x.device, requires_grad=True)

        # Scale self.A by its matrix inf-norm
        with torch.no_grad():
            if self.is_low_rank:
                self.L /= torch.linalg.matrix_norm(self.L, ord=float('inf'))
                self.L *= 0.95
                self.R /= torch.linalg.matrix_norm(self.R, ord=float('inf'))
                self.R *= 0.95
                self.A = self.L @ self.R
            else:
                self.A /= torch.linalg.matrix_norm(self.A, ord=float('inf'))
                self.A *= 0.95
                
        for _ in range(self.fixed_point_iter):
            pre_X = torch.einsum("bik,kj->bij", X, self.A[:, :]) + B_U     
        
            k = pre_X[:, :, : self.hs]
            q = pre_X[:, :, self.hs : self.hs * 2]
            v = pre_X[:, :, self.hs * 2 : self.hs * 3]
            
            if self.attention_version == 'softmax':
                wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
                wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
                wei = F.softmax(wei, dim=-1)
                out = wei @ v
            
            elif self.attention_version == 'lipschitz':
                k_norm = torch.cdist(q, k, p=2) ** 2
                wei = torch.exp(-k_norm)
                wei = wei.masked_fill(self.tril[:T, :T] == 0, 0)
                denom = 0.25 + wei.sum(dim=-1, keepdim=True)
                wei = wei / denom
                w_map = v / torch.sqrt(v ** 2 + 1)
                out = wei @ w_map
                
            else:
                raise ValueError(f"Unsupported attention version: {self.attention_version}. Choose 'softmax' or 'lipschitz'.")

            post_X = torch.cat((pre_X[:, :, : self.hs * 3], out), dim=-1)
            if torch.allclose(post_X, X, atol=1e-6):
                break
            X = post_X

        return X[:, :, self.hs * 3 :]
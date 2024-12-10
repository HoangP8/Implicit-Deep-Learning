import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    """
    One head of self-attention
    
    Attributes:
        key (nn.Linear): Linear transformation for the key.
        query (nn.Linear): Linear transformation for the query.
        value (nn.Linear): Linear transformation for the value.
        tril (torch.Tensor): Lower triangular matrix used to mask out future tokens.
        dropout (nn.Dropout): Dropout layer for regularization.
    """

    def __init__(self, head_size, n_embd, block_size, dropout, attention_version):
        """
        Initializes the attention head.

        Args:
            head_size (int): The size of the attention head (key, query, and value dimension).
            n_embd (int): Embedding dimension.
            block_size (int): Max sequence length for masking purposes.
            dropout (float): The dropout probability for attention weights.
        """

        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.attention_version = attention_version

    def forward(self, x):
        """
        Forward pass of the attention head.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, sequence_length, embedding_dim).

        Returns:
            torch.Tensor: Output of attention mechanism with shape (batch_size, sequence_length, head_size).
        """

        B, T, C = x.shape  # (batch_size, sequence_length, input_dim)
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)
        
        if self.attention_version == 'softmax':
            wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, T)
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
            wei = F.softmax(wei, dim=-1)  # (B, T, T)
            wei = self.dropout(wei)
            out = wei @ v  # (B, T, T) -> (B, T, head_size)
        
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
        
        return out


class IDLHead(nn.Module):
    """One head of self-attention implemented in IDL"""

    def __init__(self, head_size, n_embd, block_size, fixed_point_iter, attention_version, is_low_rank=False, rank=1):
        """
        Initialize the IDL self-attention head.

    Args:
        head_size (int): The size of the attention head.
        n_embd (int): Embedding dimension.
        block_size (int): The block size for masking.
        fixed_point_iter (int): The number of iterations for fixed-point optimization in DEQ.
        is_low_rank (bool, optional): Whether to use a low-rank approach. Default is False.
        k (int, optional): The rank for the low-rank approach. Default is 1.
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
        
    def forward(self, x):
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


class MultiHeadAttention(nn.Module):
    """
    Multi Head Attention mechanism.

    Attributes:
        heads (nn.ModuleList): List of attention heads.
        proj (nn.Linear): Linear transformation to project concatenated attention outputs.
        dropout (nn.Dropout): Dropout layer for regularization
    """

    def __init__(self, num_heads, head_size, n_embd, block_size, dropout, attention_version):
        """
        Initializes the multi-head attention layer.

        Args:
            num_heads (int): Number of attention heads.
            head_size (int): The size of the attention head.
            n_embd (int): Embedding dimension
            block_size (int): Max sequence length for masking purposes.
            dropout (float): The dropout probability for attention weights.
        """

        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, n_embd, block_size, dropout, attention_version) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass of multi-head attention.s

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embedding_dim).
        
        Returns:
            torch.Tensor: Output tensor with shape (batch_size, sequence_length, embedding_dim).
        """

        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """
    A simple MLP player
    """
    
    def __init__(self, n_embd, dropout):
        """
        Initializes the MLP layer.
        
        Args:
            n_embd (int): The size of the embedding dimension.
            dropout (float): The dropout probability for regularization.
        """
        
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Forward pass of the MLP network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embedding_dim).
        
        Returns:
            torch.Tensor: Output tensor of the same shape as input (batch_size, sequence_length, embedding_dim).
        """
        return self.net(x)


class Block(nn.Module):
    """
    A single GPT transformer block consisting of:
    - Layer normalization on input.
    - Multi-head attention followed by residual connection.
    - Layer normalization on output of attention.
    - Feed-forward network (MLP) followed by residual connection.
    """

    def __init__(self, n_embd, n_head, block_size, dropout, attention_version):
        """
        Initializes transformer block.
        
        Args:
            n_embd (int): Embedding dimension.
            n_head (int): Number of attention heads in multi-head attention.
            block_size (int): Max sequence length for masking purposes.
            dropout (float): The dropout probability for regularization.
        """

        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout, attention_version)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    """
    A GPT language model with transformer blocks

    Args:
        vocab_size (int): Number of unique tokens in the vocabulary.
        n_embd (int): Embedding dimension
        block_size (int): Max sequence length for the model.
        n_layer (int): Number of transformer blocks.
        n_head (int): Number of attention heads in each block.
        dropout (float): Dropout probability for regularization.
    """

    def __init__(self, vocab_size, n_embd, block_size, n_layer, n_head, dropout, attention_version):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout, attention_version) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initializes the weights of linear and embedding layers with a normal distribution.
        """

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Forward pass of the GPT language model.

        Args:
            idx (torch.Tensor): Input tensor of token indices with shape (batch_size, sequence_length).
            targets (torch.Tensor, optional): Ground truth labels for computing loss (default is None).

        Returns:
            logits (torch.Tensor): Output token logits of shape (batch_size, sequence_length, vocab_size).
            loss (torch.Tensor or None): Cross-entropy loss, or None if targets is not provided.
        """

        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, block_size):
        """
        Generate new tokens based on a given input sequence.

        Args:
            idx (torch.Tensor): Input tensor of shape (batch_size, sequence_length), the initial tokens.
            max_new_tokens (int): The maximum number of new tokens to generate.
            block_size (int): The size of the context window (max sequence length).

        Returns:
            idx (torch.Tensor): The input tensor with the generated tokens appended.
        """

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
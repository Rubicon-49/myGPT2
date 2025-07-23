import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, n_heads, qkv_bias=False):
        """
        GPT-2 style multi-head causal self-attention module.

        Args:
            d_in (int): Input embedding size (matches model embedding dimension).
            d_out (int): Output embedding size (usually same as d_in).
            context_length (int): Max number of tokens in sequence (used for causal mask).
            dropout (float): Dropout probability applied to attention weights.
            n_heads (int): Number of attention heads (e.g. 12 for GPT-2 small).
            qkv_bias (bool): Whether to include bias in Q, K, V linear projections.
        """
        super().__init__()
        assert d_out % n_heads == 0, "d_out must be divisible by n_heads"
        
        self.d_out = d_out
        self.n_heads = n_heads
        self.head_dim = d_out // n_heads
        
        # Separate projection layers for Q, K, V (required for GPT-2 compatibility)
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        
        # Final projection after concatenating all attention heads
        self.out_proj = nn.Linear(d_out, d_out)
        
        # Dropout applied to attention weights
        self.dropout = nn.Dropout(dropout)
        
        # Register a causal mask to prevent attention to future tokens
        # The diagonal=1 argument means start one diagonal above the main diagonal
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        self.register_buffer("mask", mask)
        
    def forward(self, x, return_attn_weights=False):
        """
        Compute causal multi-head attention.

        Args:
            x (Tensor): Input tensor of shape (b_size, n_tokens, d_in)

        Returns:
            Tensor of shape (b_size=batch_size, n_tokens, d_out)
        """
        b_size, n_tokens, _ = x.size()
        
        # Project input to queries, keys, and values
        q = self.W_q(x)  # (b_size, n_tokens, d_out)
        k = self.W_k(x)
        v = self.W_v(x)
        
        # Reshape for attention heads
        # (b_size, n_tokens, n_heads, head_dim)
        q = q.view(b_size, n_tokens, self.n_heads, self.head_dim)
        k = k.view(b_size, n_tokens, self.n_heads, self.head_dim)
        v = v.view(b_size, n_tokens, self.n_heads, self.head_dim) 
        
        # Transpose for attention heads
        # (b_size, n_tokens, n_heads, head_dim) → (b_size, n_heads, n_tokens, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        # (b_size, n_heads, n_tokens, head_dim) @ (b_size, n_heads, head_dim, n_tokens)
        # → (b_size, n_heads, n_tokens, n_tokens)
        att_scores = q @ k.transpose(-2, -1)
        
        # Apply causal mask (prevents attending to future postions)
        causal_mask = self.mask[:n_tokens, :n_tokens].bool()
        attn_scores = att_scores.masked_fill(causal_mask, -torch.inf)
        
        # Scale the attention scores before softmax
        # As the dimensionality of the attention heads increases, the dot products
        # (q · k) tend to grow larger in magnitude. This can make the softmax function
        # output very peaked distributions, where most of the probability mass is placed
        # on a few positions — leading to vanishing gradients and unstable training.
        #
        # To counteract this effect, we scale the dot product by the inverse square root
        # of the head dimension (i.e., divide by sqrt(d_k)), as suggested in the original
        # Transformer paper (Vaswani et al., 2017). This keeps the magnitude of the input
        # to softmax more stable regardless of dimensionality.
        attn_weights = torch.softmax(attn_scores / (self.head_dim ** 0.5), dim=-1)
        
        # Apply dropout to attention weights to prevent the model from relying too much
        # on any single attention path — helps improve generalization and reduce overfitting.
        attn_weights = self.dropout(attn_weights)
        
        # Compute the attention output for each token.
        # Each token looks at other tokens using the attention weights.
        # The values (v) are weighted by these attention scores.
        # The result is a context vector that captures information from relevant tokens.
        context_vec = attn_weights @ v  # (b_size, n_heads, n_tokens, head_dim)
        
        # Recombine heads: (b_size, n_heads, n_tokens, head_dim) → (b_size, n_tokens, d_out)
        context_vec = context_vec.transpose(1, 2)
        context_vec = context_vec.contiguous().view(b_size, n_tokens, self.d_out)
        
        # Final linear projection to combine all attention heads.
        # Each head captures different types of relationships, and their outputs are concatenated.
        # This projection maps the combined output back into the model's embedding space.
        context_vec = self.out_proj(context_vec)
        
        # Return the context vector, and optionally the attention weights for testing purposes.
        if return_attn_weights:
            return context_vec, attn_weights
        else:
            return context_vec
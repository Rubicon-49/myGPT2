import torch
import torch.nn as nn
from dataclasses import dataclass
import toml
from MultiHeadAttention import MultiHeadAttention

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        # Token embedding layer:
        # Maps each token index to a learnable embedding vector of size `d_model`.
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["d_model"])
        
        # Positional embedding layer:
        # Maps each position index to a learnable embedding vector of size `d_model`.
        self.pos_emb = nn.Embedding(cfg["context_len"], cfg["emb_dim"])
        
        # Dropout layer applied after adding token + positional embeddings.
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        # Stack of Transformer blocks.
        # Each block consists of multi-head attention, feedforward network, and layer normalization.
        self.trf_blocks = nn.ModuleList(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layer"])]
        )
        
        # Final layer nomralization
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )
    
    def forward(self, in_idx):
        # in_idx: tensor of token indices with shape [batch_size, seq_len]
        batch_size, seq_len = in_idx.shape
        
        # Look up embeddings for each token in the input sequence.
        tok_embeds = self.tok_emb(in_idx)
        
        # This represents token positions in the sequence.
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device)).unsqueeze(0)
        
        # Add token and positional embeddings.
        # Broadcasting allows pos_embeds [seq_len, emb_dim] to be added to tok_embeds.
        # Result shape: [batch_size, seq_len, emb_dim]
        x = tok_embeds + pos_embeds
        
        # Apply dropout to embeddings to prevent overfitting.
        x = self.drop_emb(x)  
        
        # Pass the embeddings through the stack of transformer blocks.
        for block in self.trf_blocks:
            x = block(x)
        
        # Apply layer normalization to stabilize the output.
        x = self.final_norm(x)
        
        # Project the final hidden states to vocabulary size logits
        logits = self.out_head(x)  # Shape [batch_size, num_tokens, vocab_size]
        return logits


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        
        # Small constant added for numerical stability when dividing by variance.
        self.eps = 1e-5
        
        # Learnable scale (gamma) parameter.
        # Initialized to all ones, so initially has no effect.
        self.scale = nn.Parameter(torch.ones(emb_dim))
        
        # Learnable shift (beta) parameter.
        # Initialized to zeros so initially has no effect.
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        
    def forward(self, x):
        # Compute the mean across the last dimension (features) for each sample.
        mean = x.mean(dim=-1, keepdim=True)
        
        # Compute the variance across the last dimension (features) for each sample.
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # Normalize: subtract mean and divide by std dev.
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        
        # Apply learned scale (gamma) and shift (beta) parameters.
        return self.scale * norm_x + self.shift

# Define a feedforward neural network block used in transformers, which expands the
# embedding size, applies a GELU activation and projects back to the orginal embedding
# size.

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], cfg["ffn_hidden_mult"] * cfg["emb_dim"]),
            nn.GELU(),
            nn.Linear(cfg["ffn_hidden_mult"] * cfg["emb_dim"], cfg["emb_dim"])
        )
    
    def forward(self, x):
        return self.layers(x)

# GELU activation function (Hendrycks and Gimpel 2016) can be implemented in several
# ways. The exact version is defined as GELU(x) = x * cumulative distribution function of
# standard normal distribution at x, which can be approximated as:

# class GELU(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.sqrt_two_over_pi = torch.sqrt(torch.tensor(2.0 / torch.pi))
        
#     def forward(self, x):
#         return 0.5 * x * (1 + torch.tanh(
#             self.sqrt_two_over_pi * (x + 0.044715 * torch.pow(x, 3))    
#         ))

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in = cfg["emb_dim"],
            d_out = cfg["emb_dim"],
            context_len=cfg["context_len"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
        
    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x) # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut # Add the original input back
        
        # Shortcut connection for feedforward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back
        
        return x
    
@dataclass
class GPTConfig:
    vocab_size: int
    emb_dim: int
    context_length: int
    n_layer: int
    n_heads: int
    ffn_hidden_mult: int
    drop_rate: float
    qkv_bias: bool
    

if __name__ == "__main__":
    config_dict = toml.load("GPT-config.toml")
    cfg = GPTConfig(**config_dict)
    model = GPTModel(cfg)
import torch
import pytest
from MultiHeadAttention import MultiHeadAttention

@pytest.fixture
def cfg():
    return {
        "d_in": 32,
        "d_out": 32,
        "b_size": 2,
        "n_tokens": 128,
        "dropout": 0.1,
        "n_heads": 4,
    }

# Test the shape consistency
def test_output_shape(cfg):
    attn = MultiHeadAttention(cfg["d_in"], cfg["d_out"], cfg["n_tokens"], cfg["dropout"], cfg["n_heads"])
    x = torch.randn(cfg["b_size"], cfg["n_tokens"], cfg["d_in"])  # batch_size=2, n_tokens=10, d_in=64
    out = attn(x)
    assert out.shape == (cfg["b_size"], cfg["n_tokens"], cfg["d_out"])

# Test whether the output of the attention layer changes if you change tokens after the current one    
def test_causal_masking(cfg):
    attn = MultiHeadAttention(cfg["d_in"], cfg["d_out"], cfg["n_tokens"], cfg["dropout"], cfg["n_heads"])
    attn.eval()
    
    x = torch.randn(cfg["b_size"], cfg["n_tokens"], cfg["d_in"])
    x_modified = x.clone()
    x_modified[:, -1, :] += 1000
    
    with torch.no_grad():
        out_original = attn(x)
        out_modified = attn(x_modified)
        
    assert torch.allclose(out_original[:, :-1, :], out_modified[:, :-1, :], atol=1e-6)
    
# This test verifies that outputs are stable in eval mode.
# It acts as a regression guard: if stochastic components like dropout
# or stateful logic are accidentally introduced, this test will catch it.
def test_deterministic_output(cfg):
    attn = MultiHeadAttention(cfg["d_in"], cfg["d_out"], cfg["n_tokens"], cfg["dropout"], cfg["n_heads"])
    attn.eval()
    x = torch.randn(cfg["b_size"], cfg["n_tokens"], cfg["d_in"])
    with torch.no_grad():
        out1 = attn(x)
        out2 = attn(x)
    assert torch.allclose(out1, out2)

# Test that both the input tensor and all model parameters receive gradients.
# Ensures end-to-end differentiability of the attention mechanism.
def test_backward_input_and_parameters(cfg):
    attn = MultiHeadAttention(cfg["d_in"], cfg["d_out"], cfg["n_tokens"], cfg["dropout"], cfg["n_heads"])
    x = torch.randn(cfg["b_size"], cfg["n_tokens"], cfg["d_in"], requires_grad=True) 
    
    out = attn(x)
    loss = out.mean()
    loss.backward()
    
    # Input gradient check
    assert x.grad is not None, "No grdient backpropagated to input tensor."
    assert torch.any(x.grad != 0), "Input tensor gradient is zero."
    
    # Parameter gradient check
    for name, param in attn.named_parameters():
        assert param.grad is not None, f"No gradient for parameter {name}."
        assert torch.any(param.grad != 0), f"Parameter {name} gradient is zero."

# Ensure that the parameter count matches expectations.
def test_parameter_count(cfg):
    
    attn = MultiHeadAttention(cfg["d_in"], cfg["d_out"], cfg["n_tokens"], cfg["dropout"], cfg["n_heads"])
    
    expected = 3 * cfg["d_in"] * cfg["d_out"] + cfg["d_in"] * cfg["d_out"]   # Q, K, V + out_proj weights
    expected += cfg["d_out"] # bias for out_proj is True and False for Q, K, V
    
    actual = sum(p.numel() for p in attn.parameters() if p.requires_grad)
    
    assert actual == expected, f"Expected {expected} parameters, but got {actual}."

@pytest.mark.parametrize("device", ["cpu", "cuda" if torch.cuda.is_available() else "cpu"])
def test_multihead_attention_device_compability(device, cfg):
    """
    Ensure the model runs correctly on both CPU and GPU (if available).
    Catches device mismatch bugs in buffers or intermediate tensors.
    """
    
    attn = MultiHeadAttention(cfg["d_in"], cfg["d_out"], cfg["n_tokens"], cfg["dropout"], cfg["n_heads"])
    attn.to(device)
    
    x = torch.randn(cfg["b_size"], cfg["n_tokens"], cfg["d_in"]).to(device)
    
    try:
        out = attn(x)
    except RuntimeError as e:
        pytest.fail(f"MultiHeadAttention failed on {device} with error: {e}")
    
    assert out.device.type == device, f"Output tensor is not on {device}."
    assert out.shape == (cfg["b_size"], cfg["n_tokens"], cfg["d_out"]), "Output shape mismatch."
    
    for param in attn.parameters():
        assert param.device.type == device, f"Parameter {param} is not on {device}."
        
    assert attn.mask.device.type == device, "Causal mask is not on the correct device."

   
@pytest.fixture
def visible_attn_weights(cfg):
    """
    Returns the attention weights tensor from a forward pass through the attention module.
    Useful for tests that verify attention distribution properties.
    """
    torch.manual_seed(42)
    attn = MultiHeadAttention(cfg["d_in"], cfg["d_out"], cfg["n_tokens"], cfg["dropout"], cfg["n_heads"])
    attn.eval()
    
    x = torch.randn(cfg["b_size"], cfg["n_tokens"], cfg["d_in"])
    with torch.no_grad():
        _, weights = attn(x, return_attn_weights=True)
    
    b_size, n_heads, n_tokens, _ = weights.shape

    causal_mask = torch.tril(torch.ones(n_tokens, n_tokens, dtype=torch.bool))
    full_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(b_size, n_heads, n_tokens, n_tokens)

    return weights, full_mask

# Check that attention weights form valid probability distributions.
# Each row of the attention matrix should sum to 1 due to softmax.
def test_attention_weights_sum_to_one(visible_attn_weights):
    weights, mask = visible_attn_weights
    masked_weights = weights * mask
    actual = masked_weights.sum(dim=-1)
    expected = torch.ones_like(actual)
    assert torch.allclose(actual, expected, atol=1e-5)
    
# Check that the entropy of each attention distribution is not too low.
# This ensures attention is not overly sharp or collapsed.
def test_attention_entropy_above_threshold(visible_attn_weights):
    weights, mask = visible_attn_weights
    masked_weights = weights * mask
    
    eps = 1e-9
    entropy = -masked_weights * torch.log(masked_weights + eps)
    entropy_sum = entropy.sum(dim=-1)
    
    visible_counts = mask.sum(dim=-1)
    valid_rows = visible_counts > 3
    
    assert torch.all(entropy_sum[valid_rows] > 1.0), "Low entropy — attention may be collapsing"
    
# Check that no single visible attention weight dominates the distribution
def test_attention_max_value_not_too_high(visible_attn_weights):
    weights, mask = visible_attn_weights
    b_size, n_heads, n_tokens, _ = weights.shape
    
    diag_mask = torch.eye(n_tokens, dtype=torch.bool)
    diag_mask = diag_mask.unsqueeze(0).unsqueeze(0).expand(b_size, n_heads, n_tokens, n_tokens)
    visible_weights = weights.masked_select(mask & ~diag_mask)

    max_val = visible_weights.max()
    assert max_val < 0.95, f"Max attention weight too high: {max_val.item():.4f}"
       
# Check that no visible attention weights are vanishingly small
    weights, mask = visible_attn_weights
    visible_weights = weights.masked_select(mask)

    min_val = visible_weights.min()
    assert min_val > 0.001, f"Min attention weight too small: {min_val.item():.6f}"

import torch

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Returns the scaled dot-product attention output.
    """
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2,-1)) /(d_k **0.5)
    attention_weights = torch.softmax(scores,dim=-1)
    output = torch.matmul(attention_weights, V)
    return output.to(torch.float32)
import torch
from typing import Optional

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Returns: attention output tensor of shape (batch, seq_q, d_v)
    """
    d_k = query.size(-1)
    scores = torch.matmul(query,key.transpose(-2,-1))
    scores = scores/ math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask,float('-inf'))
    attention = torch.softmax(scores, dim=-1)
    output = torch.matmul(attention, value)
    return output

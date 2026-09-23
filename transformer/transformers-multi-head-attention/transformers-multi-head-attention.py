import numpy as np

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Returns projected multi-head attention outputs.
    """
    batch_size, query_length, d_model = Q.shape
    _ , key_length, _ = K.shape
    d_k = d_model // num_heads

    Q_proj = Q @ W_q
    K_proj = K @ W_k
    V_proj = V @ W_v

    Q_heads = Q_proj.reshape(batch_size, query_length, num_heads, d_k).transpose(0,2,1,3)
    K_heads = K_proj.reshape(batch_size, key_length, num_heads, d_k).transpose(0,2,1,3)
    V_heads = V_proj.reshape(batch_size, key_length, num_heads, d_k).transpose(0,2,1,3)

    scores = Q_heads @ K_heads.transpose(0,1,3,2)
    scores = scores/ np.sqrt(d_k)

    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    head_outputs = attention_weights @ V_heads
    head_outputs = head_outputs.transpose(0,2,1,3)

    concatenated = head_outputs.reshape(batch_size, query_length, d_model)

    output = concatenated @ W_o
    return output.astype(np.float64)
import numpy as np

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                  W2: np.ndarray, b2: np.ndarray, gamma1: np.ndarray,
                  beta1: np.ndarray, gamma2: np.ndarray, beta2: np.ndarray,
                  num_heads: int) -> np.ndarray:
    """
    Returns the post-normalized Transformer encoder states.
    """
    x = np.asarray(x , dtype= np.float64)

    attention = multi_head_attention(x,x,x, W_q, W_k, W_v, W_o, num_heads)

    z = layer_norm (x+attention, gamma1, beta1, eps=1e-6)

    ffn = feed_forward(z, W1, b1, W2, b2)
    y = layer_norm(z+ffn, gamma2, beta2, eps=1e-6)
    return y.astype(np.float64)
def multi_head_attention(
    Q: np.ndarray,
    K:np.ndarray,
    V: np.ndarray,
    W_q: np.ndarray,
    W_k: np.ndarray,
    W_v: np.ndarray,
    W_o: np.ndarray,
    num_heads: int 
) -> np.ndarray:
    batch_size, query_length, d_model = Q.shape
    key_length = K.shape[1]

    d_k = d_model // num_heads

    Q_proj = Q @ W_q
    K_proj = Q @ W_k
    V_proj = Q @ W_v

    Q_heads = Q_proj.reshape(batch_size, query_length, num_heads, d_k). transpose(0,2,1,3)
    K_heads = K_proj.reshape(batch_size, key_length, num_heads, d_k).transpose(0,2,1,3)
    V_heads = V_proj.reshape(batch_size, key_length, num_heads, d_k).transpose(0,2,1,3)

    scores = ( Q_heads @ K_heads.transpose(0,1,3,2)) / np.sqrt(d_k)

    scores = scores - np.max(scores, axis=-1, keepdims= True)

    weights = np.exp(scores)

    weights /= np.sum(weights, axis=-1, keepdims=True)
    heads = weights @ V_heads

    heads = heads.transpose(0,2,1,3)
    concatenated = heads.reshape(batch_size, query_length, d_model)

    return concatenated @ W_o

def feed_forward(
    x: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray
) -> np.ndarray:
    hidden = x@ w1 + b1
    hidden = np.maximum(0.0, hidden)

    return hidden @ w2 + b2

def layer_norm(
    x: np.ndarray,
    gamma : np.ndarray,
    beta : np.ndarray,
    eps : float = 1e-6,
) -> np.ndarray:

    mean = np.mean(x, axis=-1, keepdims=True)

    variancs = np.mean ((x-mean)**2,
                       axis=-1,
                       keepdims=True
                       )
    normalized = (
        (x-mean)/ np.sqrt(variancs + eps)
    )
    return gamma * normalized + beta
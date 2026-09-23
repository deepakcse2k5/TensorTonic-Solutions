import numpy as np

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns the last-axis normalized array.
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)

    normalize = (x-mean)/ np.sqrt(variance + eps)
    output = gamma * normalize + beta
    return output.astype(np.float64)
import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0.0

    for _ in range(steps):
        logits = X @ weights + bias
        predictions = _sigmoid(logits)

        error = predictions - y
        dw = (1/n_samples) * ( X.T @ error)
        db = np.mean(error)
        weights -= lr * dw
        bias -= lr*db
    return weights, bias
    
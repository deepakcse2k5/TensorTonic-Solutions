import numpy as np

def softmax_regression(X: list, y: list, num_classes: int, lr: float, n_iters: int) -> tuple:
    """
    Returns the fitted weight matrix and bias vector.
    """
    X = np.asarray(X, dtype = float)
    y = np.asarray(y, dtype = int)

    n_samples, n_features = X.shape

    weights = np.zeros((n_features, num_classes))
    bias = np.zeros(num_classes)

    Y = np.zeros((n_samples, num_classes))
    Y[np.arange(n_samples), y] = 1
    for _ in range(n_iters):
        z = X @ weights + bias

        z = z - np.max(z, axis=1,keepdims=True)
        exp_z = np.exp(z)
        P = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        error = P - Y
        dw = (1/n_samples) * (X.T @ error)
        db = (1/n_samples) * np.sum(error, axis=0)

        weights -= lr*dw
        bias -= lr*db

    return weights.tolist(), bias.tolist()
        

import numpy as np

def svm_hinge_sgd(X: list, y: list, lr: float, lam: float, n_epochs: int) -> dict:
    """
    Returns fitted parameters and training predictions.
    """
    X = np.asarray(X, dtype= np.float64)
    y = np.asarray(y, dtype= np.float64)

    n_samples, n_features = X.shape

    weights = np.zeros(n_features, dtype=np.float64)
    bias = np.float64(0.0)

    for _ in range(n_epochs):
        for i in range(n_samples):
            xi = X[i]
            yi = y[i]
            margin = yi * (np.dot(xi, weights) + bias)
            if margin < 1:
                weights -= lr * (lam * weights - yi*xi)
                bias += lr * yi
            else:
                weights -= lr * lam * weights
    scores = X @ weights + bias
    predictions = np.where(scores > 0, 1, -1)

    weights = np.round(weights, 4).tolist()
    bias = float(np.round(bias,4))

    return {
        "weights":weights,
        "bias": bias,
        "predictions": predictions.astype(int).tolist()
    }

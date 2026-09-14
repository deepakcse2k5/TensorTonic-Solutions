def ridge_regression(X, y, lr, epochs, alpha):
    """
    Perform ridge regression using gradient descent.
    Returns: tuple of (weights_list, bias)
    """
    X = np.asarray(X, dtype = float)
    y = np.asarray(y, dtype = float)

    n_samples, n_features = X.shape

    weights = np.zeros(n_features)
    bias = 0.0

    for _ in range(epochs):
        y_pred = X @ weights + bias
        error = y_pred - y
        dw = (2/n_samples) * (X.T @ error) +  2* alpha* weights
        db = (2/n_samples) * (np.sum(error))

        weights -= lr*dw
        bias -= lr * db
    return weights.tolist(), float(bias)
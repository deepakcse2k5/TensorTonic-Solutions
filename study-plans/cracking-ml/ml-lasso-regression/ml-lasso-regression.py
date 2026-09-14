def lasso_regression(X, y, lr, epochs, alpha):
    """
    Perform Lasso Regression using gradient descent with L1 subgradient.
    Returns: tuple of (weights_list, bias_float)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    n_sample, n_feature = X.shape

    weights = np.zeros(n_feature)
    bias = 0.0

    for _ in range(epochs):
        y_pred = X @ weights + bias
        error = y_pred - y

        dw = (2/n_sample) * (X.T @ error) + alpha * np.sign(weights)
        db = (2/n_sample) * (np.sum(error))
        weights -= lr * dw
        bias -= lr * db
    return weights.tolist(), float(bias)
import numpy as np

def averaged_perceptron(X_train, y_train, X_test, n_epochs=10):
    """
    Returns: A list of predicted labels (-1 or +1) for each test point
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype = int)
    X_test = np.asarray(X_test, dtype = float)

    n_samples, n_features = X_train.shape

    weights = np.zeros(n_features)
    bias = 0.0

    weighted_sum = np.zeros(n_features)
    bias_sum = 0.0
    total_steps = 0

    for _ in range(n_epochs):
        for x, y in zip(X_train, y_train):
            score = np.dot(weights, x) + bias
            if y * score <=0:
                weights += y*x
                bias += y
            weighted_sum += weights
            bias_sum += bias
            total_steps += 1
    avg_weight = weighted_sum / total_steps
    avg_bias = bias_sum / total_steps
    scores = X_test @ avg_weight + avg_bias
    predictions = np.where(scores>0, 1, -1)
    return predictions.tolist()

import numpy as np

def knn_classify(X_train, y_train, X_test, k=3):
    """
    Returns: A list of predicted integer labels for each test point
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype = int)
    X_test = np.asarray(X_test, dtype=float)
    predictions = []

    for x in X_test:
        distances = np.sqrt(np.sum((X_train-x)**2, axis=1))
        nearest_indices = np.argsort(distances)[:k]

        nearest_label = y_train[nearest_indices]

        counts = np.bincount(nearest_label)
        prediction = np.argmax(counts)
        predictions.append(int(prediction))

    return predictions
        
        

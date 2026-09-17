import numpy as np

def gaussian_nb(X_train, y_train, X_test):
    """
    Returns: A list of predicted integer labels for each test point
    """
    X_train = np.asarray(X_train, dtype = float)
    y_train = np.asarray(y_train, dtype = int)
    X_test = np.asarray(X_test, dtype = float)

    predictions = []
    classes = np.unique(y_train)

    for x in X_test:
        class_log_probs = []
        for c in classes:
            x_c = X_train[y_train==c]
            mean = np.mean(x_c, axis=0)
            if len(x_c) > 1:
                var = np.var(x_c, axis=0, ddof=1)
            else:
                var = np.zeros(x_c.shape[1])

            var = np.maximum(var, 1e-9)

            prior = len(x_c) / len(X_train)
            log_likelihood = -0.5 * np.sum(np.log(2*np.pi*var) + ((x-mean)**2)/ var)
            log_posterior = np.log(prior) + log_likelihood
            class_log_probs.append(log_posterior)
        predictions.append(int(classes[np.argmax(class_log_probs)]))
    return predictions
            

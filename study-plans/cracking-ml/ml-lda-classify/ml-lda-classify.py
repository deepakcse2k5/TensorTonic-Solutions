import numpy as np

def lda_classify(X_train: list, y_train: list, X_test: list) -> list:
    """
    Returns one predicted label for each test row using LDA.
    """

    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test, dtype=float)

    classes = np.unique(y_train)
    n_samples, n_features = X_train.shape
    n_classes = len(classes)

    # Class means and priors
    means = {}
    priors = {}

    for c in classes:
        X_c = X_train[y_train == c]

        means[c] = np.mean(X_c, axis=0)
        priors[c] = len(X_c) / n_samples

    # Pooled shared covariance matrix
    covariance = np.zeros((n_features, n_features))

    for c in classes:
        X_c = X_train[y_train == c]
        centered = X_c - means[c]

        covariance += centered.T @ centered

    covariance /= (n_samples - n_classes)

    # Pseudo-inverse for numerical stability
    inv_covariance = np.linalg.pinv(covariance)

    predictions = []

    for x in X_test:

        scores = []

        for c in classes:
            mean = means[c]

            score = (
                x @ inv_covariance @ mean
                - 0.5 * mean @ inv_covariance @ mean
                + np.log(priors[c])
            )

            scores.append(score)

        scores = np.asarray(scores)

        # Normal LDA decision
        max_score = np.max(scores)

        # Check for exact/tiny numerical tie
        tied = np.isclose(scores, max_score, atol=1e-12)

        if np.sum(tied) > 1:
            # Tie-break using nearest class mean
            distances = [
                np.linalg.norm(x - means[c])
                if tied[i] else np.inf
                for i, c in enumerate(classes)
            ]

            prediction = classes[np.argmin(distances)]

        else:
            prediction = classes[np.argmax(scores)]

        predictions.append(prediction.item())

    return predictions
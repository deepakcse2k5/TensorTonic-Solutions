import numpy as np
from fractions import Fraction

def cart_classify(X_train: list, y_train: list, X_test: list, max_depth: int = 5, min_samples: int = 2) -> list:
    """
    Returns one predicted class label per test row.
    """
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=int)

    def purity_score(labels):
        _, counts = np.unique(labels, return_counts=True)

        probabilities = counts / len(labels)

        return Fraction(
            sum(int(count) ** 2 for count in counts), len(labels)
        )

    def build(rows, depth):
        labels = y[rows]

        classes, counts = np.unique(labels, return_counts = True)

        majority = int(classes[np.argmax(counts)])

        if (depth >= max_depth or len(rows) < min_samples or len(classes)==1):
            return majority

        best_score  = purity_score(labels)
        best_split = None

        for feature in range(X.shape[1]):
            values = X[rows, feature]
            for threshold in np.unique(values):
                mask = values <= threshold
                left_rows = rows[mask]
                right_rows = rows[~mask]

                if len(left_rows) ==0 or len(right_rows)==0:
                    continue 
                score = (
                    purity_score(y[left_rows])
                    + purity_score(y[right_rows])
                )

                if score > best_score:
                    best_score = score
                    best_split = (feature, threshold, left_rows, right_rows)

                
        if best_split is None:
            return majority

        feature, threshold, left_rows, right_rows = best_split
        return (
            feature,
            threshold,
            build(left_rows, depth+1),
            build(right_rows, depth+1)
        )
    tree = build(np.arange(len(y)), 0)
    predictions = []
    for row in X_test:
        node = tree
        while isinstance(node, tuple):
            feature, threshold, left, right = node
            node = left if row[feature] <= threshold else right
        predictions.append(int(node))
    return predictions
                
            
            

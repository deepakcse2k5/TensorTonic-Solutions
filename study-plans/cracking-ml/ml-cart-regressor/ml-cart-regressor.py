import numpy as np
from fractions import Fraction

def cart_regress(X_train: list, y_train: list, X_test: list, max_depth: int = 5, min_samples: int = 2) -> list:
    """
    Returns one regression prediction per test row.
    """
    X = np.asarray(X_train, dtype=float)
    y  = [Fraction(str(value)) for value in y_train]

    def build(rows, depth):
        targets = [y[i] for i in rows]

        total = sum(targets, Fraction(0))

        mean = total / len(rows)

        if (depth >= max_depth or len(rows)< min_samples or all(value == targets[0] for value in targets)):
            return float(mean)

        best_score = total * total / len(rows)
        best_split = None

        for feature in range(X.shape[1]):
            values = X[rows, feature]

            for threshold in np.unique(values):
                mask = values <= threshold
                left_rows = rows[mask]
                right_rows = rows[~mask]

                if len(left_rows)==0 or len(right_rows)==0:
                    continue 

                left_sum = sum((y[i] for i in left_rows), Fraction(0))
                right_sum = total - left_sum
                score = (
                    left_sum * left_sum / len(left_rows)
                    + right_sum * right_sum/ len(right_rows)
                )
                if score > best_score:
                    best_score = score
                    best_split = (
                        feature, threshold, left_rows, right_rows
                    )
        if best_split is None:
            return float(mean)
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
        predictions.append(round(float(node), 4))
    return predictions
    
                
        

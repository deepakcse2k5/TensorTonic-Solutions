import numpy as np

def permutation_importance(X: list, y: list, predict_fn, n_repeats: int = 5, seed: int = 42) -> list:
    """
    Returns one permutation importance per feature.
    """
    X = np.asarray(X)
    y = np.array(y)

    baseline_pred = np.asarray(predict_fn(X))
    baseline_acc = np.mean(baseline_pred==y)


    rng = np.random.RandomState(seed)
    n_features = X.shape[1]
    importances = []

    for j in range(n_features):
        total_drops = 0
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[:,j] = X_perm[rng.permutation(len(X)), j]

            perm_pred = np.asarray(predict_fn(X_perm))
            perm_acc = np.mean(perm_pred==y)

            total_drops += baseline_acc - perm_acc

        importance = total_drops / n_repeats
        importances.append(round(float(importance),4))
    return importances
    

from typing import Dict

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


def train_svm(X_train, y_train, X_test, y_test) -> Dict:
    """Train a SVM model on the given datasets.

    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        X_test (np.ndarray): Testing features
        y_test (np.ndarray): Testing labels

    Returns:
        Dict: Dictionary containing the training and testing accuracies
    """
    # Grid search parameters
    param_grid = {"kernel": ["rbf"], "C": [1.0, 10.0], "gamma": ["scale", 0.1]}

    # Train with grid search
    svm = SVC(cache_size=4000)
    cv = 3 if len(X_train) < 10000 else 2

    grid_search = GridSearchCV(
        svm, param_grid, cv=cv, n_jobs=-1, pre_dispatch="2*n_jobs", error_score="raise"
    )

    grid_search.fit(X_train, y_train)
    best_svm = grid_search.best_estimator_

    # Calculate accuracies
    train_accuracy = best_svm.score(X_train, y_train) * 100.0
    test_accuracy = best_svm.score(X_test, y_test) * 100.0

    return {
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
    }

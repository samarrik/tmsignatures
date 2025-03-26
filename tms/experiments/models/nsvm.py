from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.svm import OneClassSVM


class MultiNoveltySVM:
    """
    Multi-class novelty detection using one-class SVMs.

    This class trains a separate one-class SVM for each class in the dataset,
    with hyperparameter optimization to find the best parameters for each class.
    The SVMs are used to identify whether new samples belong to known classes
    or are novelties (outliers).
    """

    def __init__(
        self,
        kernels_options: List[str],
        nus_options: List[float],
        gammas_options: List[Union[float, str]],
    ):
        """
        Initialize the MultiNoveltySVM model.

        Args:
            kernels_options: List of kernel types to try (e.g., ['rbf', 'linear'])
            nus_options: List of nu values to try (controls the fraction of outliers)
            gammas_options: List of gamma values to try (kernel coefficient for 'rbf')
        """
        self.svms: List[Optional[OneClassSVM]] = []
        self.classes: Optional[np.ndarray] = None
        self.kernels_options = kernels_options
        self.nus_options = nus_options
        self.gammas_options = gammas_options
        self.best_params: List[Dict[str, Any]] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultiNoveltySVM":
        """
        Fit one-class SVM for each class with hyperparameter optimization.

        For each class, finds the best hyperparameters by evaluating models on
        a validation set of inliers (samples from the class) and outliers
        (samples from other classes).

        Args:
            X: Training samples of shape (n_samples, n_features)
            y: Class labels of shape (n_samples,)

        Returns:
            self: The fitted model

        Raises:
            ValueError: If input data is invalid
        """
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")

        if len(X) == 0:
            raise ValueError("Empty training set")

        # Initialize class attributes
        self.classes = np.unique(y)
        self.svms = [None] * len(self.classes)
        self.best_params = [None] * len(self.classes)

        # Train a separate SVM for each class
        for i, class_label in enumerate(self.classes):
            # Extract samples for this class
            X_class = X[y == class_label]

            # Find best hyperparameters for this class
            best_params, best_score = self._find_best_params(
                X_class, X[y != class_label]
            )
            self.best_params[i] = best_params

            # Train final SVM with best parameters
            self.svms[i] = OneClassSVM(**best_params)
            self.svms[i].fit(X_class)

        return self

    def _find_best_params(
        self, X_inliers: np.ndarray, X_outliers: np.ndarray
    ) -> Tuple[Dict[str, Any], float]:
        """
        Find best hyperparameters for a one-class SVM.

        Evaluates different hyperparameter combinations by measuring
        how well they score inliers highly and outliers lowly.

        Args:
            X_inliers: Samples from the target class
            X_outliers: Samples from other classes

        Returns:
            Tuple of (best parameters dict, best score)
        """
        best_score = -float("inf")
        best_params = None

        # Balance the outlier set if it's much larger than the inlier set
        if len(X_outliers) > len(X_inliers):
            indices = np.random.choice(
                len(X_outliers),
                size=min(len(X_inliers), len(X_outliers)),
                replace=False,
            )
            X_outliers = X_outliers[indices]

        # Grid search for best parameters
        for kernel in self.kernels_options:
            for nu in self.nus_options:
                for gamma in self.gammas_options:
                    # Create and evaluate temporary SVM
                    score, params = self._evaluate_model_params(
                        X_inliers, X_outliers, kernel, nu, gamma
                    )

                    if score > best_score:
                        best_score = score
                        best_params = params

        return best_params, best_score

    def _evaluate_model_params(
        self,
        X_inliers: np.ndarray,
        X_outliers: np.ndarray,
        kernel: str,
        nu: float,
        gamma: Union[float, str],
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate a specific set of hyperparameters.

        Creates a temporary SVM with the given parameters and evaluates
        how well it separates inliers from outliers.

        Args:
            X_inliers: Samples from the target class
            X_outliers: Samples from other classes
            kernel: Kernel type
            nu: Nu parameter value
            gamma: Gamma parameter value

        Returns:
            Tuple of (score, parameters dict)
        """
        # Create and fit temporary SVM
        params = {"kernel": kernel, "nu": nu, "gamma": gamma}
        try:
            temp_svm = OneClassSVM(**params)
            temp_svm.fit(X_inliers)

            # Score inliers (should be high)
            inlier_score = temp_svm.decision_function(X_inliers).mean()

            # Score outliers (should be low)
            outlier_score = -temp_svm.decision_function(X_outliers).mean()

            # Combined score (high = good separation)
            score = inlier_score + outlier_score

            return score, params
        except Exception as e:
            # If model fails to fit, return very low score
            print(f"Error evaluating parameters {params}: {e}")
            return -float("inf"), params

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy score.

        Args:
            X: Test samples of shape (n_samples, n_features)
            y: True class labels of shape (n_samples,)

        Returns:
            Accuracy as a float between 0.0 and 1.0

        Raises:
            ValueError: If model hasn't been fitted yet
        """
        if not self._is_fitted():
            raise ValueError("Model has not been fitted yet.")

        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Each sample is assigned to the class whose SVM gives
        the highest decision function score (least anomalous).

        Args:
            X: Test samples of shape (n_samples, n_features)

        Returns:
            Predicted class labels of shape (n_samples,)

        Raises:
            ValueError: If model hasn't been fitted yet
        """
        if not self._is_fitted():
            raise ValueError("Model has not been fitted yet.")

        # Get decision scores from all SVMs
        scores = np.array([svm.decision_function(X) for svm in self.svms])

        # Return class with highest score (least anomalous)
        return self.classes[np.argmax(scores, axis=0)]

    def _is_fitted(self) -> bool:
        """
        Check if the model has been fitted.

        Returns:
            True if the model has been fitted, False otherwise
        """
        return self.svms is not None and len(self.svms) > 0 and self.classes is not None

    def get_best_params(self) -> List[Dict[str, Any]]:
        """
        Get the best parameters found for each class.

        Returns:
            List of parameter dictionaries, one for each class

        Raises:
            ValueError: If model hasn't been fitted yet
        """
        if not self._is_fitted():
            raise ValueError("Model has not been fitted yet.")

        return self.best_params


def train_nsvm(X_train, y_train, X_test, y_test) -> Dict:
    """Train a NSVM model on the given datasets.

    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        X_test (np.ndarray): Testing features
        y_test (np.ndarray): Testing labels

    Returns:
        Dict: Dictionary containing the training and testing accuracies
    """
    # Train model
    n_svm = MultiNoveltySVM(
        kernels_options=["rbf"],
        nus_options=[0.01, 0.05, 0.1],
        gammas_options=["scale", 0.1],
    )
    n_svm.fit(X_train, y_train)

    # Calculate accuracies
    train_accuracy = n_svm.score(X_train, y_train) * 100.0
    test_accuracy = n_svm.score(X_test, y_test) * 100.0

    return {
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
    }

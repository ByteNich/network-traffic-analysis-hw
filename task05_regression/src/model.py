"""
Linear regression model for salary prediction.
"""
from pathlib import Path
from typing import Optional

import numpy as np


class LinearRegressionModel:
    """Simple linear regression model with L2 regularization (Ridge)."""

    def __init__(self, alpha: float = 1.0) -> None:
        """
        Initialize the model.

        Args:
            alpha: L2 regularization strength.
        """
        self.alpha = alpha
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionModel":
        """
        Fit the model to training data.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target vector of shape (n_samples,).

        Returns:
            Self for method chaining.
        """
        # Normalize features
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1  # Avoid division by zero

        X_normalized = (X - self.mean) / self.std

        # Add bias term
        n_samples = X_normalized.shape[0]
        X_with_bias = np.column_stack([np.ones(n_samples), X_normalized])

        # Closed-form solution with L2 regularization (Ridge regression)
        # w = (X^T X + alpha * I)^(-1) X^T y
        n_features = X_with_bias.shape[1]
        regularization = self.alpha * np.eye(n_features)
        regularization[0, 0] = 0  # Don't regularize bias term

        XTX = X_with_bias.T @ X_with_bias
        XTy = X_with_bias.T @ y

        weights_with_bias = np.linalg.solve(XTX + regularization, XTy)

        self.bias = weights_with_bias[0]
        self.weights = weights_with_bias[1:]

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict salary values.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predicted salaries as numpy array.
        """
        if self.weights is None:
            raise ValueError("Model is not fitted. Call fit() first.")

        # Normalize using training statistics
        X_normalized = (X - self.mean) / self.std

        predictions = X_normalized @ self.weights + self.bias

        # Ensure non-negative predictions
        predictions = np.maximum(predictions, 0)

        return predictions

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate R² score.

        Args:
            X: Feature matrix.
            y: True target values.

        Returns:
            R² score (coefficient of determination).
        """
        predictions = self.predict(X)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        return 1 - (ss_res / ss_tot)

    def save(self, path: Path) -> None:
        """
        Save model weights to file.

        Args:
            path: Path to save the model.
        """
        if self.weights is None:
            raise ValueError("Model is not fitted. Call fit() first.")

        model_data = {
            "weights": self.weights,
            "bias": self.bias,
            "mean": self.mean,
            "std": self.std,
            "alpha": self.alpha,
        }
        np.save(path, model_data, allow_pickle=True)

    def load(self, path: Path) -> "LinearRegressionModel":
        """
        Load model weights from file.

        Args:
            path: Path to the saved model.

        Returns:
            Self for method chaining.
        """
        model_data = np.load(path, allow_pickle=True).item()

        self.weights = model_data["weights"]
        self.bias = model_data["bias"]
        self.mean = model_data["mean"]
        self.std = model_data["std"]
        self.alpha = model_data["alpha"]

        return self

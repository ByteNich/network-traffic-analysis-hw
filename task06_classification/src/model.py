"""Классификатор уровня разработчика (junior/middle/senior)."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DeveloperLevelClassifier:
    """Мультиклассовая логистическая регрессия для предсказания уровня разработчика."""

    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()
        self.model = LogisticRegression(C=C, max_iter=max_iter, multi_class="multinomial", solver="lbfgs")
        self.classes_ = []

    def fit(self, X: np.ndarray, y: list):
        X_scaled = self.scaler.fit_transform(X)
        y_encoded = self.encoder.fit_transform(y)
        self.classes_ = list(self.encoder.classes_)
        self.model.fit(X_scaled, y_encoded)
        return self

    def predict(self, X: np.ndarray) -> list:
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        return list(self.encoder.inverse_transform(y_pred))

    def report(self, X: np.ndarray, y_true: list) -> str:
        y_pred = self.predict(X)
        return classification_report(y_true, y_pred, target_names=sorted(set(y_true)))

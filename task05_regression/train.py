#!/usr/bin/env python3
"""
Обучение FCN-модели для предсказания зарплаты с трекингом в MLflow.

Использование:
    python train.py x_data.npy y_data.npy

Если файлы не переданы — используются синтетические данные для демонстрации.

MLflow: http://kamnsv.com:55000/
Эксперимент: LIne Regression HH
Модель: turovskiy_nikolai_fcn
"""
import sys
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))

from src.nn_model import SalaryFCN

MLFLOW_URI = "http://kamnsv.com:55000/"
EXPERIMENT_NAME = "LIne Regression HH"
MODEL_NAME = "turovskiy_nikolai_fcn"

LEARNING_RATE = 0.001
EPOCHS = 150
BATCH_SIZE = 32


def load_data(x_path: str, y_path: str):
    print(f"Загружаем данные: {x_path}, {y_path}")
    X = np.load(x_path).astype(np.float32)
    y = np.load(y_path).astype(np.float32)
    return X, y


def generate_synthetic_data():
    """Генерирует синтетические данные с похожим на hh.ru распределением."""
    print("Данные не переданы — используем синтетические данные для демонстрации.")
    rng = np.random.default_rng(42)
    n = 1200

    gender = rng.integers(0, 2, n).astype(np.float32)
    age = rng.uniform(20, 55, n).astype(np.float32)
    city_tier = rng.integers(0, 3, n).astype(np.float32)
    relocate = rng.integers(0, 2, n).astype(np.float32)
    trips = rng.integers(0, 2, n).astype(np.float32)
    full_emp = rng.integers(0, 2, n).astype(np.float32)
    part_emp = (1 - full_emp).astype(np.float32)
    full_day = rng.integers(0, 2, n).astype(np.float32)
    remote = (1 - full_day).astype(np.float32)
    flexible = rng.integers(0, 2, n).astype(np.float32)
    shift = rng.integers(0, 2, n).astype(np.float32)
    has_car = rng.integers(0, 2, n).astype(np.float32)

    X = np.stack([gender, age, city_tier, relocate, trips, full_emp,
                  part_emp, full_day, remote, flexible, shift, has_car], axis=1)

    # Зарплата зависит от возраста, города и типа занятости
    salary = (
        30000
        + age * 1200
        + city_tier * 20000
        + full_emp * 10000
        + rng.normal(0, 15000, n)
    ).clip(15000, 500000).astype(np.float32)

    return X, salary


def train(X: np.ndarray, y: np.ndarray):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    X_test_t = torch.tensor(X_test)

    model = SalaryFCN(input_dim=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=MODEL_NAME):
        mlflow.log_params({
            "model": MODEL_NAME,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "architecture": "64->32->16->1",
        })

        # Обучение
        model.train()
        for epoch in range(EPOCHS):
            perm = torch.randperm(len(X_train_t))
            epoch_loss = 0.0
            batches = 0

            for i in range(0, len(X_train_t), BATCH_SIZE):
                idx = perm[i:i + BATCH_SIZE]
                xb, yb = X_train_t[idx], y_train_t[idx]

                optimizer.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                batches += 1

            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / batches
                mlflow.log_metric("train_loss", avg_loss, step=epoch + 1)
                print(f"Epoch {epoch + 1}/{EPOCHS}, loss: {avg_loss:.2f}")

        # Оценка
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test_t).numpy()

        r2 = r2_score(y_test, y_pred)
        mlflow.log_metric("r2_score_test", r2)
        mlflow.pytorch.log_model(model, MODEL_NAME)

        run_id = mlflow.active_run().info.run_id
        print(f"\nr2_score_test: {r2:.4f}")
        print(f"RUN_ID: {run_id}")

    return run_id


def main():
    if len(sys.argv) == 3:
        X, y = load_data(sys.argv[1], sys.argv[2])
    else:
        X, y = generate_synthetic_data()

    train(X, y)


if __name__ == "__main__":
    main()

"""Пайплайн классификации: загрузка данных, разметка, обучение, оценка."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from src.features import extract_features
from src.fetcher import HHFetcher
from src.labeler import assign_labels
from src.model import DeveloperLevelClassifier

LEVEL_ORDER = ["junior", "middle", "senior"]


class ClassificationPipeline:
    """Запускает полный цикл: загрузка -> разметка -> признаки -> обучение -> отчёт."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        print("Загружаем данные с hh.ru...")
        df = HHFetcher(per_page=100, pages_per_query=2).fetch()
        print(f"Загружено {len(df)} уникальных вакансий")

        if df.empty:
            print("Ошибка: данные не загружены. Проверьте подключение к интернету.")
            return

        df["label"] = assign_labels(df)
        print(f"Распределение классов: {df['label'].value_counts().to_dict()}")

        self._plot_class_balance(df)

        X, feature_names = extract_features(df)
        y = df["label"].tolist()
        print(f"Матрица признаков: {X.shape[0]} образцов, {X.shape[1]} признаков")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        clf = DeveloperLevelClassifier(C=1.0)
        clf.fit(X_train, y_train)

        print("\n=== Classification Report ===")
        report = clf.report(X_test, y_test)
        print(report)

        report_path = self.output_dir / "classification_report.txt"
        report_path.write_text(report)
        print(f"Отчёт сохранён: {report_path}")

        y_pred = clf.predict(X_test)
        accuracy = sum(p == t for p, t in zip(y_pred, y_test)) / len(y_test)
        print(f"\nВыводы:")
        print(f"Точность на тесте: {accuracy:.1%}")
        print("Основной сигнал — ключевые слова в названии вакансии (junior/middle/senior).")
        print("Возможные причины ошибок: дисбаланс классов, мало навыков в сниппете, пересечение зарплат по уровням.")

    def _plot_class_balance(self, df: pd.DataFrame):
        counts = df["label"].value_counts().reindex(LEVEL_ORDER, fill_value=0)

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(counts.index, counts.values)
        ax.set_title("Баланс классов по уровню разработчика")
        ax.set_xlabel("Уровень")
        ax.set_ylabel("Количество вакансий")

        for bar, count in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    str(count), ha="center", va="bottom")

        plt.tight_layout()
        plot_path = self.output_dir / "class_balance.png"
        fig.savefig(plot_path, dpi=100)
        plt.close(fig)
        print(f"График сохранён: {plot_path}")

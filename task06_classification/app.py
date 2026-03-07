#!/usr/bin/env python3
"""
Классификатор уровня IT-разработчика (junior/middle/senior) — PoC.

Загружает вакансии с hh.ru API, обучает логистическую регрессию,
выводит classification report и график баланса классов.

Использование:
    python app.py [output_dir]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import ClassificationPipeline


def main():
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    ClassificationPipeline(output_dir=output_dir).run()


if __name__ == "__main__":
    main()

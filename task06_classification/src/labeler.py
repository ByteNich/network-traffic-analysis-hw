"""Разметка вакансий по уровню: junior / middle / senior."""
import pandas as pd

SENIOR_KEYWORDS = ["senior", "сеньор", "лид", "lead", "principal", "architect", "архитект", "head of", "team lead"]
JUNIOR_KEYWORDS = ["junior", "джун", "trainee", "intern", "стажер", "начинающий"]
MIDDLE_KEYWORDS = ["middle", "мидл"]

# Если ключевых слов в названии нет — используем уровень опыта
EXPERIENCE_MAP = {
    "noExperience": "junior",
    "between1And3": "junior",
    "between3And6": "middle",
    "moreThan6": "senior",
}


def assign_labels(df: pd.DataFrame) -> pd.Series:
    """Присваивает метку junior/middle/senior каждой вакансии."""
    return df.apply(_label_row, axis=1)


def _label_row(row) -> str:
    title = str(row.get("title", "")).lower()

    if any(kw in title for kw in SENIOR_KEYWORDS):
        return "senior"
    if any(kw in title for kw in JUNIOR_KEYWORDS):
        return "junior"
    if any(kw in title for kw in MIDDLE_KEYWORDS):
        return "middle"

    return EXPERIENCE_MAP.get(str(row.get("experience_id", "")), "middle")

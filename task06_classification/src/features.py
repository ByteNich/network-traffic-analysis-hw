"""Извлечение числовых признаков из датафрейма вакансий."""
import numpy as np
import pandas as pd

TIER1_CITIES = ["москва", "санкт-петербург", "питер"]
TIER2_CITIES = [
    "новосибирск", "екатеринбург", "казань", "нижний новгород",
    "челябинск", "самара", "красноярск", "пермь", "воронеж", "краснодар",
]

SKILL_KEYWORDS = [
    "python", "java", "javascript", "typescript", "golang", "c++", "c#",
    "rust", "php", "ruby", "kotlin", "swift", "scala",
    "sql", "postgresql", "mysql", "mongodb", "redis",
    "docker", "kubernetes", "aws", "linux", "git",
    "react", "vue", "angular", "django", "spring", "fastapi",
]


def extract_features(df: pd.DataFrame):
    """Строит матрицу числовых признаков из датафрейма вакансий."""
    features = pd.DataFrame(index=df.index)

    features["salary_mid"] = df.apply(_salary_mid, axis=1)
    features["city_tier"] = df["city"].apply(_city_tier)
    features["is_remote"] = (df["schedule_id"] == "remote").astype(int)
    features["is_fulltime"] = (df["employment_id"] == "full").astype(int)

    text = (df["title"] + " " + df["requirement"]).str.lower()
    for skill in SKILL_KEYWORDS:
        col_name = f"skill_{skill.replace('+', 'p').replace('#', 's')}"
        features[col_name] = text.str.contains(skill, regex=False).astype(int)

    return features.values.astype(np.float64), list(features.columns)


def _salary_mid(row) -> float:
    s_from = float(row.get("salary_from") or 0)
    s_to = float(row.get("salary_to") or 0)
    if s_from > 0 and s_to > 0:
        return (s_from + s_to) / 2.0
    return s_from or s_to


def _city_tier(city: str) -> int:
    city_lower = str(city).lower()
    if any(c in city_lower for c in TIER1_CITIES):
        return 2
    if any(c in city_lower for c in TIER2_CITIES):
        return 1
    return 0

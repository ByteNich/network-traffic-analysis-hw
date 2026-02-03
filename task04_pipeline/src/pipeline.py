"""
Pipeline for processing hh.ru resume data using Chain of Responsibility pattern.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd


class Handler(ABC):
    """Abstract handler for Chain of Responsibility pattern."""

    def __init__(self) -> None:
        self._next_handler: Optional[Handler] = None

    def set_next(self, handler: Handler) -> Handler:
        """Set the next handler in the chain."""
        self._next_handler = handler
        return handler

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process data and pass to next handler if exists."""
        df = self.process(df)
        if self._next_handler:
            return self._next_handler.handle(df)
        return df

    @abstractmethod
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process the dataframe. Must be implemented by subclasses."""
        pass


class DropColumnsHandler(Handler):
    """Remove unnecessary columns."""

    COLUMNS_TO_DROP = [
        "Unnamed: 0",
        "Ищет работу на должность:",
        "Опыт (двойное нажатие для полной версии)",
        "Последенее/нынешнее место работы",
        "Последеняя/нынешняя должность",
        "Образование и ВУЗ",
        "Обновление резюме",
    ]

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop unnecessary columns."""
        cols_to_drop = [col for col in self.COLUMNS_TO_DROP if col in df.columns]
        return df.drop(columns=cols_to_drop)


class SalaryHandler(Handler):
    """Parse and clean salary data."""

    # Exchange rates (approximate, for historical data ~2019)
    EXCHANGE_RATES = {
        "руб.": 1.0,
        "USD": 65.0,
        "EUR": 73.0,
        "KZT": 0.17,
        "UAH": 2.5,
        "BYN": 32.0,
        "UZS": 0.0075,
        "AZN": 38.0,
        "KGS": 0.93,
        "грн.": 2.5,
    }

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse salary and convert to rubles."""
        df = df.copy()
        df["salary"] = df["ЗП"].apply(self._parse_salary)
        df = df.drop(columns=["ЗП"])
        # Remove rows with invalid salary
        df = df[df["salary"] > 0]
        df = df[df["salary"] < 1_000_000]  # Filter outliers
        return df

    def _parse_salary(self, value: str) -> float:
        """Extract salary value and convert to rubles."""
        if pd.isna(value) or not isinstance(value, str):
            return 0.0

        value = value.strip()
        if not value:
            return 0.0

        # Extract numeric part
        numbers = re.findall(r"[\d\s]+", value)
        if not numbers:
            return 0.0

        try:
            amount = float(numbers[0].replace(" ", "").replace("\xa0", ""))
        except ValueError:
            return 0.0

        # Determine currency and convert
        for currency, rate in self.EXCHANGE_RATES.items():
            if currency in value:
                return amount * rate

        return amount


class GenderAgeHandler(Handler):
    """Parse gender and age from combined column."""

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract gender and age."""
        df = df.copy()
        col_name = "Пол, возраст"

        df["gender"] = df[col_name].apply(self._extract_gender)
        df["age"] = df[col_name].apply(self._extract_age)
        df = df.drop(columns=[col_name])

        # Remove rows with invalid age
        df = df[df["age"] > 0]
        df = df[df["age"] < 100]

        return df

    def _extract_gender(self, value: str) -> int:
        """Extract gender: 1 for male, 0 for female."""
        if pd.isna(value):
            return -1
        value = str(value).lower()
        if "мужчина" in value:
            return 1
        if "женщина" in value:
            return 0
        return -1

    def _extract_age(self, value: str) -> int:
        """Extract age from string."""
        if pd.isna(value):
            return 0
        match = re.search(r"(\d+)\s*(?:год|лет|года)", str(value))
        if match:
            return int(match.group(1))
        return 0


class CityHandler(Handler):
    """Encode city information."""

    # Major cities with higher salaries
    TIER1_CITIES = ["москва", "санкт-петербург", "питер"]
    TIER2_CITIES = [
        "новосибирск", "екатеринбург", "казань", "нижний новгород",
        "челябинск", "самара", "омск", "ростов-на-дону", "уфа",
        "красноярск", "пермь", "воронеж", "волгоград", "краснодар",
    ]

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode city tier and relocation readiness."""
        df = df.copy()
        col_name = "Город"

        df["city_tier"] = df[col_name].apply(self._get_city_tier)
        df["ready_to_relocate"] = df[col_name].apply(self._check_relocation)
        df["ready_for_business_trips"] = df[col_name].apply(self._check_business_trips)
        df = df.drop(columns=[col_name])

        return df

    def _get_city_tier(self, value: str) -> int:
        """Get city tier: 2 for Moscow/SPb, 1 for major cities, 0 for others."""
        if pd.isna(value):
            return 0
        value = str(value).lower()
        city = value.split(",")[0].strip()

        if any(c in city for c in self.TIER1_CITIES):
            return 2
        if any(c in city for c in self.TIER2_CITIES):
            return 1
        return 0

    def _check_relocation(self, value: str) -> int:
        """Check if ready to relocate."""
        if pd.isna(value):
            return 0
        value = str(value).lower()
        if "не готов к переезду" in value or "не готова к переезду" in value:
            return 0
        if "готов к переезду" in value or "готова к переезду" in value:
            return 1
        return 0

    def _check_business_trips(self, value: str) -> int:
        """Check if ready for business trips."""
        if pd.isna(value):
            return 0
        value = str(value).lower()
        if "не готов к командировкам" in value or "не готова к командировкам" in value:
            return 0
        if "готов к командировкам" in value or "готова к командировкам" in value:
            return 1
        if "готов к редким командировкам" in value or "готова к редким командировкам" in value:
            return 1
        return 0


class EmploymentHandler(Handler):
    """Parse employment type."""

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode employment type."""
        df = df.copy()
        col_name = "Занятость"

        df["full_employment"] = df[col_name].apply(
            lambda x: 1 if pd.notna(x) and "полная" in str(x).lower() else 0
        )
        df["partial_employment"] = df[col_name].apply(
            lambda x: 1 if pd.notna(x) and "частичная" in str(x).lower() else 0
        )
        df = df.drop(columns=[col_name])

        return df


class ScheduleHandler(Handler):
    """Parse work schedule."""

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode schedule type."""
        df = df.copy()
        col_name = "График"

        df["full_day"] = df[col_name].apply(
            lambda x: 1 if pd.notna(x) and "полный день" in str(x).lower() else 0
        )
        df["remote_work"] = df[col_name].apply(
            lambda x: 1 if pd.notna(x) and "удаленн" in str(x).lower() else 0
        )
        df["flexible_schedule"] = df[col_name].apply(
            lambda x: 1 if pd.notna(x) and "гибкий" in str(x).lower() else 0
        )
        df["shift_work"] = df[col_name].apply(
            lambda x: 1 if pd.notna(x) and ("сменный" in str(x).lower() or "вахт" in str(x).lower()) else 0
        )
        df = df.drop(columns=[col_name])

        return df


class AutoHandler(Handler):
    """Parse car ownership."""

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode car ownership."""
        df = df.copy()
        col_name = "Авто"

        df["has_car"] = df[col_name].apply(
            lambda x: 1 if pd.notna(x) and "имеется" in str(x).lower() else 0
        )
        df = df.drop(columns=[col_name])

        return df


class FinalCleanupHandler(Handler):
    """Final cleanup and validation."""

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove any remaining NaN values and validate data."""
        df = df.copy()

        # Fill remaining NaN with 0
        df = df.fillna(0)

        # Ensure all columns are numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Remove rows with invalid gender
        df = df[df["gender"] >= 0]

        return df


class DataPipeline:
    """Main pipeline class that chains all handlers."""

    def __init__(self) -> None:
        """Initialize the pipeline with all handlers."""
        self.first_handler = DropColumnsHandler()

        # Build the chain
        (
            self.first_handler
            .set_next(SalaryHandler())
            .set_next(GenderAgeHandler())
            .set_next(CityHandler())
            .set_next(EmploymentHandler())
            .set_next(ScheduleHandler())
            .set_next(AutoHandler())
            .set_next(FinalCleanupHandler())
        )

    def process(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Process the dataframe through the pipeline.

        Returns:
            Tuple of (x_data, y_data) as numpy arrays.
        """
        processed_df = self.first_handler.handle(df)

        # Separate features and target
        y_data = processed_df["salary"].values
        x_data = processed_df.drop(columns=["salary"]).values

        return x_data.astype(np.float64), y_data.astype(np.float64)

    def get_feature_names(self, df: pd.DataFrame) -> list[str]:
        """Get feature names after processing (for debugging)."""
        processed_df = self.first_handler.handle(df)
        return [col for col in processed_df.columns if col != "salary"]

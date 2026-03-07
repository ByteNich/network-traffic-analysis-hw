"""Загрузка вакансий IT-разработчиков с hh.ru API."""
import time

import pandas as pd
import requests

HH_API_URL = "https://api.hh.ru/vacancies"

SEARCH_QUERIES = [
    "junior разработчик",
    "junior developer",
    "middle разработчик",
    "middle developer",
    "senior разработчик",
    "senior developer",
    "программист junior",
    "программист senior",
    "джуниор разработчик",
    "сеньор разработчик",
]


class HHFetcher:
    """Загружает вакансии IT-разработчиков с hh.ru."""

    def __init__(self, per_page: int = 100, pages_per_query: int = 2):
        self.per_page = min(per_page, 100)
        self.pages_per_query = pages_per_query
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "hw06-research/1.0"

    def fetch(self) -> pd.DataFrame:
        """Загружает вакансии по всем запросам, убирает дубликаты."""
        records = []
        seen_ids = set()

        for query in SEARCH_QUERIES:
            for page in range(self.pages_per_query):
                batch = self._fetch_page(query, page)
                for item in batch:
                    if item["id"] not in seen_ids:
                        seen_ids.add(item["id"])
                        records.append(item)
                time.sleep(0.25)

        return pd.DataFrame(records)

    def _fetch_page(self, query: str, page: int) -> list:
        params = {
            "text": query,
            "specialization": "1",
            "per_page": self.per_page,
            "page": page,
            "only_with_salary": True,
            "currency": "RUR",
        }
        try:
            response = self.session.get(HH_API_URL, params=params, timeout=15)
            response.raise_for_status()
            return [self._parse_item(item) for item in response.json().get("items", [])]
        except requests.RequestException as e:
            print(f"Warning: не удалось загрузить ({query}, page {page}): {e}")
            return []

    def _parse_item(self, item: dict) -> dict:
        salary = item.get("salary") or {}
        experience = item.get("experience") or {}
        return {
            "id": str(item.get("id", "")),
            "title": item.get("name", ""),
            "experience_id": experience.get("id", ""),
            "salary_from": float(salary.get("from") or 0),
            "salary_to": float(salary.get("to") or 0),
            "city": (item.get("area") or {}).get("name", ""),
            "schedule_id": (item.get("schedule") or {}).get("id", ""),
            "employment_id": (item.get("employment") or {}).get("id", ""),
            "requirement": str((item.get("snippet") or {}).get("requirement", "") or ""),
        }

from enum import Enum


class ProcessMode(Enum):
    DOUBLE = 1
    STRINGIFY = 2
    FILTER_EVEN = 3


def process_items(items: list, mode: ProcessMode, log: bool = False) -> list | None:
    """Process list items according to mode. Returns None if input is empty."""
    if not items:
        print("empty!")
        return None

    result = []
    for item in items:
        if mode == ProcessMode.DOUBLE:
            result.append(item * 2)
        elif mode == ProcessMode.STRINGIFY:
            result.append(str(item))
        elif mode == ProcessMode.FILTER_EVEN:
            if item % 2 == 0:
                result.append(item)

        if log:
            print(f"item: {item}, result: {result}")

    return result

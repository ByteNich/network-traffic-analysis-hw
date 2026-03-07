def scale_array(array: list, lower_bound: float, upper_bound: float) -> list:
    """Scale values to [lower_bound, upper_bound] range (min-max normalization)."""
    if not array:
        return []

    min_val = min(array)
    max_val = max(array)
    span = max_val - min_val

    if span == 0:
        return [lower_bound] * len(array)

    return [
        lower_bound + (x - min_val) / span * (upper_bound - lower_bound)
        for x in array
    ]

#!/usr/bin/env python3
"""
Salary prediction application.

Usage:
    python app.py path/to/x_data.npy

Returns:
    List of predicted salaries in rubles (float).
"""
import json
import sys
from pathlib import Path

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.model import LinearRegressionModel


def main() -> None:
    """Load model and predict salaries from input data."""
    if len(sys.argv) != 2:
        print("Usage: python app.py path/to/x_data.npy", file=sys.stderr)
        sys.exit(1)

    x_data_path = Path(sys.argv[1])

    if not x_data_path.exists():
        print(f"Error: File not found: {x_data_path}", file=sys.stderr)
        sys.exit(1)

    # Load input data
    x_data = np.load(x_data_path)

    # Load model
    resources_dir = Path(__file__).parent / "resources"
    model_path = resources_dir / "model_weights.npy"

    if not model_path.exists():
        print(f"Error: Model weights not found at {model_path}", file=sys.stderr)
        print("Please run train.py first to train the model.", file=sys.stderr)
        sys.exit(1)

    model = LinearRegressionModel()
    model.load(model_path)

    # Predict
    predictions = model.predict(x_data)

    # Output as list of floats
    salary_list = predictions.tolist()

    # Print as JSON for easy parsing
    print(json.dumps(salary_list))


if __name__ == "__main__":
    main()

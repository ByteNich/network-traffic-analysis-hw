#!/usr/bin/env python3
"""
Data processing pipeline application.

Usage:
    python app_pipeline.py path/to/hh.csv

Creates x_data.npy and y_data.npy in the same directory as the input file.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import DataPipeline


def main() -> None:
    """Process CSV and create numpy arrays."""
    if len(sys.argv) != 2:
        print("Usage: python app_pipeline.py path/to/hh.csv", file=sys.stderr)
        sys.exit(1)

    csv_path = Path(sys.argv[1])

    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    # Load data
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path, encoding="utf-8")
    print(f"Loaded {len(df)} rows")

    # Process through pipeline
    print("Processing data through pipeline...")
    pipeline = DataPipeline()
    x_data, y_data = pipeline.process(df)

    print(f"Processed data shape: X={x_data.shape}, y={y_data.shape}")

    # Save output files in the same directory as input
    output_dir = csv_path.parent
    x_path = output_dir / "x_data.npy"
    y_path = output_dir / "y_data.npy"

    np.save(x_path, x_data)
    np.save(y_path, y_data)

    print(f"Saved: {x_path}")
    print(f"Saved: {y_path}")

    # Print feature info
    feature_names = pipeline.get_feature_names(df)
    print(f"\nFeatures ({len(feature_names)}):")
    for i, name in enumerate(feature_names):
        print(f"  {i}: {name}")


if __name__ == "__main__":
    main()

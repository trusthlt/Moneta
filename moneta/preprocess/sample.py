"""
Step 3: sample up to 50 records per NACE section (random_state=42), as for MONETA.
Reads <paths.processed>/nace_combined_Gold.json, writes <paths.processed>/MONETA_sample.json
"""

import argparse
from pathlib import Path

import pandas as pd

from moneta.utils import (
    get_logger,
    load_config,
    read_json_list,
    resolve_path,
    write_json,
)

logger = get_logger()


def sample_sections(records: list[dict], n: int = 50, seed: int = 42) -> list[dict]:
    """Per section (sorted), min(n, size) records via pandas' sample(random_state=seed)."""
    df = pd.DataFrame({"category": [r["category"] for r in records]})
    index = [
        i
        for _, group in df.groupby("category")
        for i in group.sample(min(len(group), n), random_state=seed).index
    ]
    return [records[i] for i in index]


def main(input_path=None, output_path=None, n: int = 50, seed: int = 42) -> Path:
    processed = resolve_path(load_config()["paths"]["processed"])
    input_path = resolve_path(input_path or processed / "nace_combined_Gold.json")
    output_path = resolve_path(output_path or processed / "MONETA_sample.json")
    sample = sample_sections(read_json_list(input_path), n, seed)
    write_json(sample, output_path)
    logger.info(f"Sampled {len(sample)} records to {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.input, args.output, args.n, args.seed)

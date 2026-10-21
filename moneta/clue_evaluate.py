"""
Clue analysis for multi-turn runs (Fig. 3/5, Table 11): per clue source
- IDR: information discovery rate, 1 - (#"no economic activity" clues / #valid clues)
- Correctness: mean share of a clue's [keywords] that point to the gold section
- Effectiveness: mean share of a clue's [keywords] that point to the predicted section
Writes <input_dir>/clue_evaluation_results.csv for every MultiTurn run folder.
"""

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd

from moneta.constants import NACE_LABELS, NACE_SECTIONS, NACE_TITLES_TO_KEYWORDS
from moneta.evaluate import get_attributes_from_folder_name
from moneta.utils import get_logger, read_json_list, resolve_path

logger = get_logger()

CLUE_COLUMNS = [
    "osm_clues",
    "satellite_clues",
    "wikidata_clues",
    "wikipedia_clues",
    "website_clues",
]
LABELS = NACE_LABELS + ["UNK"]  # matrix columns, in this order
TITLE_TO_LABEL = {title: label for label, title in NACE_SECTIONS.items()}
# Keywords matched case-insensitively (clues are lowercased), e.g. [IT] -> J
KEYWORD_TO_LABEL = {
    keyword.lower(): TITLE_TO_LABEL[title]
    for title, keywords in NACE_TITLES_TO_KEYWORDS.items()
    for keyword in keywords
}


def is_valid_clue(clue) -> bool:
    """Missing, empty, 'None' and 'cuda OOM' clues are not counted (the original counted '' and OOM)."""
    return bool(clue) and clue not in ("None", "cuda OOM")


def load_clue_result(path):
    rows = []
    for r in read_json_list(path):
        history = r.get("history") or {}
        rows.append(
            {
                "name": r.get("name", ""),
                "category": r["category"],
                "llm_response": r.get("llm_response", ""),
                **{col: history.get(col) for col in CLUE_COLUMNS},
            }
        )
    return pd.DataFrame(rows)


def generate_keywords_from_clue(clue):
    if not is_valid_clue(clue):
        return []
    return re.findall(r"\[(.*?)\]", clue.lower())


def generate_frequency_vectors(clue_list):
    """Row-normalised keyword counts per section (columns in LABELS order)."""
    matrix = np.zeros((len(clue_list), len(LABELS)), dtype=float)
    for i, clue in enumerate(clue_list):
        for keyword in generate_keywords_from_clue(clue):
            if keyword in KEYWORD_TO_LABEL:
                matrix[i, LABELS.index(KEYWORD_TO_LABEL[keyword])] += 1
        row_sum = matrix[i].sum()
        if row_sum > 0:
            matrix[i] /= row_sum
    return matrix


def get_counts(df):
    """Per clue source: #valid clues and #'no economic activity' clues."""
    valid = [int(df[col].map(is_valid_clue).sum()) for col in CLUE_COLUMNS]
    no_activity = [
        int(df[col].str.lower().str.contains("no economic activity", na=False).sum())
        for col in CLUE_COLUMNS
    ]
    return valid, no_activity


def label_scores(frequency_matrices: dict, labels, valid_counts):
    """Mean frequency mass on the given label per clue source (UNK for unknown labels)."""
    indices = np.array(
        [
            LABELS.index(str(x).strip().upper())
            if str(x).strip().upper() in LABELS
            else len(LABELS) - 1
            for x in labels
        ]
    )
    rows = np.arange(len(indices))
    return [
        frequency_matrices[col][rows, indices].sum() / n if n > 0 else 0
        for col, n in zip(CLUE_COLUMNS, valid_counts, strict=True)
    ]


def evaluation_pipeline(path):
    df = load_clue_result(path)
    valid, no_activity = get_counts(df)
    frequencies = {
        col: generate_frequency_vectors(df[col].tolist()) for col in CLUE_COLUMNS
    }
    correctness = label_scores(frequencies, df["category"].tolist(), valid)
    effectiveness = label_scores(frequencies, df["llm_response"].tolist(), valid)
    results = {}
    for i, col in enumerate(CLUE_COLUMNS):
        source = col.removesuffix("_clues")
        results[f"IDR_{source}"] = 1 - no_activity[i] / valid[i] if valid[i] > 0 else 0
        results[f"Correctness_{source}"] = correctness[i]
        results[f"Effectiveness_{source}"] = effectiveness[i]
    return results


def main(input_dir) -> Path:
    input_dir = resolve_path(input_dir)
    rows = []
    for folder in sorted(input_dir.glob("MultiTurn_*")):
        if (folder / "inference_results.json").exists():
            rows.append(
                {
                    **get_attributes_from_folder_name(folder.name),
                    **evaluation_pipeline(folder / "inference_results.json"),
                }
            )
    if not rows:
        raise FileNotFoundError(f"No MultiTurn_*/inference_results.json in {input_dir}")
    save_path = input_dir / "clue_evaluation_results.csv"
    pd.DataFrame(rows).to_csv(save_path, index=False)
    logger.info(f"Evaluated clues of {len(rows)} runs; saved to {save_path}")
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Folder with MultiTurn_* run folders",
    )
    main(parser.parse_args().input_dir)

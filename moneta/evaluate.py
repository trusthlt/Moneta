"""
Accuracy, macro precision/recall/F1 and unknown ratio for every run folder in --results
(Tables 2-3). Writes <results>/evaluation_results.csv; --plot adds per-run confusion matrices.
Prediction normalisation is unchanged from the paper runs.
"""

import argparse
import json
from pathlib import Path
import re

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from moneta.constants import NACE_LABELS
from moneta.utils import (
    get_logger,
    load_config,
    parse_folder_name,
    read_json_list,
    resolve_path,
)

logger = get_logger()


def parse_explanation(explanation, correction=True):
    """LLM_RESPONSE from a JSON block in the explanation (used for parse_err responses)."""
    match = re.search(r"{.*}", explanation, re.DOTALL)
    if match is None:
        return "UNK" if correction else ""
    try:
        parsed_json = json.loads(match.group(0))
    except json.JSONDecodeError:
        return "UNK" if correction else ""
    return parsed_json.get("LLM_RESPONSE", "UNK" if correction else "")


def process_predictions(results):
    """Valid single-label answers are kept; 'assistant\\n' answers take the last part;
    parse_err answers are re-parsed from the explanation; everything else is UNK."""
    new_predictions = []
    for r in results:
        exp = r.get("llm_explanation", "")
        pred = r.get("llm_response", "")
        if len(pred.split()) == 1 and pred.split()[0] in NACE_LABELS:
            new_predictions.append(pred)
        elif "assistant\n" in pred:
            new_predictions.append(pred.split("assistant\n")[-1].strip())
        elif "parse_err" in pred:
            llm_response = parse_explanation(exp)
            new_predictions.append(
                llm_response if llm_response in NACE_LABELS else "UNK"
            )
        else:
            new_predictions.append("UNK")
    return new_predictions


def calculate_unk_out(results):
    """Number of explicit 'UNK' answers (the model abstained)."""
    unk_out = 0
    for r in results:
        exp = r.get("llm_explanation", "")
        pred = r.get("llm_response", "")
        if "assistant\n" in pred:
            pred = pred.split("assistant\n")[-1].strip()
        elif "parse_err" in pred:
            pred = parse_explanation(exp, correction=False)
        if pred == "UNK":
            unk_out += 1
    return unk_out


def process_results(results):
    y_true = [str(x["category"]) for x in results]
    [x["llm_response"] for x in results]  # fail loudly on missing responses
    y_pred = [str(x) for x in process_predictions(results)]
    labels = sorted(set(y_true + y_pred))
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return acc, precision, recall, f1, labels, conf_matrix


def save_conf_matrix(conf_matrix, labels, output_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.DataFrame(conf_matrix, index=labels, columns=labels)
    df.to_csv(output_path.with_suffix(".csv"))
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(df.values, cmap="Blues")
    for (i, j), v in pd.DataFrame(df.values).stack().items():
        ax.text(j, i, int(v), ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set(title="Confusion Matrix", xlabel="Predicted", ylabel="Actual")
    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".png"))
    plt.close(fig)


def map_sources(sources: list):
    if not sources:
        return "None"
    if "satellite" in sources and "osm" in sources and "source" in sources:
        return "All"
    return "+".join(sorted(sources))


def map_model_name_size(model_name: str):
    match = re.search(r"(\d+)([BM])$", model_name)
    if match:
        return model_name[: match.start()].strip().replace(" ", "."), match.group(0)
    return model_name, "Unknown"


def get_attributes_from_folder_name(folder_name):
    attributes = parse_folder_name(folder_name)
    base_name, size = map_model_name_size(attributes["model_name"])
    return {
        "Pipeline": attributes["pipeline"],
        "Task": attributes["task"],
        "Prompt Config": attributes["prompt_config"],
        "Output Format": attributes["output_format"],
        "Model Name": base_name,
        "Model Size": size,
        "Data Sources": map_sources(attributes["data_resources"]),
    }


def read_all_inference_results(results_dir: Path, plot: bool = False):
    """One row per <results_dir>/<run name>/inference_results.json (folders without one are skipped)."""
    data = []
    for folder in sorted(
        p for p in results_dir.iterdir() if (p / "inference_results.json").exists()
    ):
        attributes = get_attributes_from_folder_name(folder.name)
        results = read_json_list(folder / "inference_results.json")
        acc, precision, recall, f1, labels, conf_matrix = process_results(results)
        if plot:
            save_conf_matrix(conf_matrix, labels, folder / "confusion_matrix")
        attributes.update(
            {
                "Accuracy": acc,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "Unknown Ratio": calculate_unk_out(results) / len(results),
            }
        )
        data.append(attributes)
    return data


def main(results_dir=None, plot=False) -> Path:
    results_dir = resolve_path(results_dir or load_config()["paths"]["results"])
    df = pd.DataFrame(read_all_inference_results(results_dir, plot))
    if df.empty:
        raise FileNotFoundError(f"No */inference_results.json in {results_dir}")
    save_path = results_dir / "evaluation_results.csv"
    df.to_csv(save_path, index=False)
    logger.info(f"Evaluated {len(df)} runs; saved to {save_path}")
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=str, default=None, help="Default: paths.results"
    )
    parser.add_argument(
        "--plot", action="store_true", help="Save confusion matrices per run"
    )
    args = parser.parse_args()
    main(args.results, args.plot)

import json

import pandas as pd
import pytest

from moneta import clue_evaluate, evaluate
from moneta.constants import NACE_LABELS
from moneta.utils import parse_folder_name


def test_process_predictions_normalisation():
    results = [
        {"llm_response": "C"},  # valid label
        {"llm_response": "system\n...\nassistant\nG"},  # chat-template leak: last part
        {
            "llm_response": "parse_err",
            "llm_explanation": '{"LLM_RESPONSE": "I"}',
        },  # re-parsed
        {"llm_response": "parse_err", "llm_explanation": "cuda OOM"},  # JSON-mode OOM
        {"llm_response": "UNK"},
        {"llm_response": "The answer is C"},  # free text
    ]
    assert evaluate.process_predictions(results) == ["C", "G", "I", "UNK", "UNK", "UNK"]
    assert evaluate.calculate_unk_out(results) == 1


def test_process_results_metrics():
    results = [
        {"category": c, "llm_response": p}
        for c, p in [("A", "A"), ("A", "B"), ("B", "B"), ("C", "x")]
    ]
    acc, _, _, _, labels, conf = evaluate.process_results(results)
    assert acc == 0.5  # A, B correct; B wrong; "x" -> UNK
    assert labels == ["A", "B", "C", "UNK"]
    assert conf.sum() == 4


@pytest.mark.parametrize("record", [{"llm_response": "A"}, {"category": "A"}])
def test_missing_fields_fail_loudly(record):
    with pytest.raises(KeyError):
        evaluate.process_results([record])


def test_run_name_round_trip():
    attrs = evaluate.get_attributes_from_folder_name(
        "ZeroShot_NACE_Extended_JSON_InternVL3-38B_osm_satellite_source"
    )
    assert attrs["Pipeline"] == "ZeroShot"
    assert (attrs["Model Name"], attrs["Model Size"], attrs["Data Sources"]) == (
        "InternVL3",
        "38B",
        "All",
    )
    assert (
        parse_folder_name("ZeroShot_NACE_Simple_Text_Qwen2-5-VL-7B_none")[
            "data_resources"
        ]
        == []
    )


def test_evaluate_main(tmp_path):
    run = tmp_path / "ZeroShot_NACE_Simple_Text_Qwen2-5-VL-7B_satellite"
    run.mkdir()
    (run / "inference_results.json").write_text(
        json.dumps(
            [
                {"category": "A", "llm_response": "A"},
                {"category": "B", "llm_response": "UNK"},
            ]
        )
    )
    (tmp_path / "limit10").mkdir()  # folders without results are skipped
    df = pd.read_csv(evaluate.main(tmp_path, plot=True))
    assert df.loc[0, "Accuracy"] == 0.5 and df.loc[0, "Unknown Ratio"] == 0.5
    assert (run / "confusion_matrix.png").exists()


def test_clue_columns_follow_label_order():
    # H (transport) and I (accommodation) were swapped in the original column order
    freq = clue_evaluate.generate_frequency_vectors(
        ["[transport]", "[hotels]", "[IT] and [telecom]"]
    )
    assert freq[0, NACE_LABELS.index("H")] == 1
    assert freq[1, NACE_LABELS.index("I")] == 1
    assert freq[2, NACE_LABELS.index("J")] == 1  # [IT] now matches


def test_clue_metrics(tmp_path):
    run = tmp_path / "MultiTurn_NACE_Simple_Text_Qwen2-5-VL-7B_osm_satellite_source"
    run.mkdir()
    records = [
        {
            "category": "A",
            "llm_response": "A",
            "history": {
                "satellite_clues": "[crops] [hotels]",
                "osm_clues": "No economic activity clues found.",
            },
        },
        {
            "category": "I",
            "llm_response": "G",
            "history": {"satellite_clues": "[hotels]", "osm_clues": "cuda OOM"},
        },
    ]
    (run / "inference_results.json").write_text(json.dumps(records))
    row = pd.read_csv(clue_evaluate.main(tmp_path)).iloc[0]
    assert row["Correctness_satellite"] == pytest.approx((0.5 + 1) / 2)
    assert row["Effectiveness_satellite"] == pytest.approx((0.5 + 0) / 2)
    assert row["IDR_osm"] == 0  # 1 valid clue (OOM excluded), with no activity
    assert row["IDR_satellite"] == 1

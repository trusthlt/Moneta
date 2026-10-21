"""Pipelines end-to-end with a fake model (no GPU / torch needed)."""

import json

from moneta import clue_generator, multiturn, zero_shot
from moneta.schema.data import (
    HistorySchema,
    ImagePathsSchema,
    MultimodalInputSchema,
    SourceSchema,
)
from moneta.schema.experiment import ExperimentConfigSchema, ModelConfigSchema


class FakeModel:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def run(self, images, texts, prompt):
        self.calls.append((images, texts, prompt))
        return [self.response] * len(images or texts)


def exp_config(output_format="Text", resources=()):
    return ExperimentConfigSchema(
        model_settings=ModelConfigSchema(model_name="Qwen2.5-VL-7B", batch_size=2),
        output_format=output_format,
        data_resources=list(resources),
    )


def test_load_batch_data():
    item = MultimodalInputSchema(
        name="Bakery",
        image_paths=ImagePathsSchema(satellite_path="sat.png"),
        sources=SourceSchema(website="We bake bread", wikidata="None"),
    )
    images, texts = zero_shot.load_batch_data([item])
    assert images == [{"satellite": "sat.png"}]
    # Literal "None" strings (missing Wikidata in the extended data) are kept, as in the paper runs
    assert texts == ["Wikidata: None\nWebsite: We bake bread\nName: Bakery"]


def test_zero_shot_end_to_end(tmp_path, monkeypatch):
    model = FakeModel('```json\n{"LLM_RESPONSE": "C", "EXPLANATION": "factory"}\n```')
    monkeypatch.setattr(zero_shot, "get_model", lambda settings: model)
    path = zero_shot.main(exp_config("JSON"), output_dir=tmp_path, limit=3)
    assert path.parent.name == "ZeroShot_NACE_Simple_JSON_Qwen2-5-VL-7B_none"
    assert path.parent.parent.name == "limit3"  # never overwrites a full run
    run_config = json.loads((path.parent / "run_config.json").read_text())
    assert (
        run_config["limit"] == 3 and run_config["experiment"]["output_format"] == "JSON"
    )
    results = json.loads(path.read_text())
    assert len(results) == 3 and len(model.calls) == 2  # batch_size 2
    assert (
        results[0]["llm_response"] == "C" and results[0]["llm_explanation"] == "factory"
    )
    assert {"id", "type", "name", "category", "llm_response"} <= results[0].keys()


def test_cuda_oom_json_matches_paper_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(zero_shot, "get_model", lambda settings: FakeModel("cuda OOM"))
    path = zero_shot.main(exp_config("JSON"), output_dir=tmp_path, limit=1)
    result = json.loads(path.read_text())[0]
    assert (result["llm_response"], result["llm_explanation"]) == (
        "parse_err",
        "cuda OOM",
    )


def test_generate_clue_texts_archived_format():
    item = MultimodalInputSchema(
        name="Farm",
        history=HistorySchema(
            osm_clues="No economic activity clues found.",
            satellite_clues="[crops]",
            website_clues="cuda OOM",
        ),
    )
    assert multiturn.generate_clue_texts([item]) == [
        "osm_clues: No economic activity clues found.\nsatellite_clues: [crops]\nName: Farm\n"
    ]


def test_multiturn_end_to_end(tmp_path, monkeypatch):
    # Archived clue-file format: clues under a top-level "clues" dict
    record = {
        "id": 1,
        "type": "way",
        "bbox": [0, 0, 0, 0],
        "category": "A",
        "name": "Farm",
        "osm_tags": {"name": "Farm"},
        "clues": {"satellite": "[crops]"},
    }
    clue_path = tmp_path / "clues.json"
    clue_path.write_text(json.dumps([record]))
    model = FakeModel("A")
    monkeypatch.setattr(multiturn, "get_model", lambda settings: model)
    config = exp_config("Text", ["osm", "satellite", "source"])
    path = multiturn.main(config, clue_path=clue_path, output_dir=tmp_path)
    assert (
        path.parent.name
        == "MultiTurn_NACE_Simple_Text_Qwen2-5-VL-7B_osm_satellite_source"
    )
    assert model.calls[0][1] == ["satellite_clues: [crops]\nName: Farm\n"]
    assert json.loads(path.read_text())[0]["llm_response"] == "A"


def test_process_modality_skips_missing_inputs():
    items = [
        MultimodalInputSchema(
            sources=SourceSchema(website="text"), image_paths=ImagePathsSchema()
        ),
        MultimodalInputSchema(sources=SourceSchema(), image_paths=ImagePathsSchema()),
    ]
    model = FakeModel("clue")
    assert clue_generator.process_modality("website", items, model, "p") == [
        "clue",
        None,
    ]
    assert model.calls == [(None, ["text"], "p")]
    assert clue_generator.process_modality("osm", items, model, "p") == [None, None]

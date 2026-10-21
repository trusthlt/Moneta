import json

import pytest

from moneta.dataloader import moneta as loader
from moneta.dataloader.moneta import MonetaDataset

RAW = {"id": 1, "type": "way", "bbox": [1.0, 2.0, 3.0, 4.0], "category": "A"}


def raw_record(**tags):
    return {**RAW, "tags": {"name": "Test", **tags}}


def enriched_record():
    # Fetched text in tags, original links top-level (format of MONETA_extended.json)
    record = raw_record(
        wikidata="Wikidata text", wikipedia="Wikipedia text", website="Website text"
    )
    record.update(wikidata="Q42", wikipedia="en:Test", website="https://example.com")
    return record


@pytest.fixture
def setup(tmp_path, monkeypatch):
    """Write records to tmp files, point the config at them and create the images."""

    def _setup(records, images=True):
        (tmp_path / "osm").mkdir(exist_ok=True)
        (tmp_path / "sat").mkdir(exist_ok=True)
        if images:
            for r in records:
                (tmp_path / "osm" / f"{r['type']}_{r['id']}.png").touch()
                (tmp_path / "sat" / f"{r['type']}_{r['id']}.png").touch()
        (tmp_path / "data.json").write_text(json.dumps(records))
        config = {
            "paths": {
                "data": str(tmp_path / "data.json"),
                "data_extended": str(tmp_path / "data.json"),
                "osm_images": str(tmp_path / "osm"),
                "satellite_images": str(tmp_path / "sat"),
            }
        }
        monkeypatch.setattr(loader, "load_config", lambda: config)
        return tmp_path

    return _setup


def test_all_resources(setup):
    tmp = setup([enriched_record()])
    item = MonetaDataset(data_resources=["osm", "satellite", "source"]).data[0]
    assert (item.id, item.type, item.name, item.category) == (1, "way", "Test", "A")
    assert item.bbox == [1.0, 2.0, 3.0, 4.0]
    assert item.image_paths.osm_path == str(tmp / "osm" / "way_1.png")
    assert item.image_paths.satellite_path == str(tmp / "sat" / "way_1.png")
    assert item.sources.wikidata == "Wikidata text"
    assert item.sources.website == "Website text"


def test_unrequested_resources_are_empty(setup):
    setup([enriched_record()])
    item = MonetaDataset(data_resources=["osm"]).data[0]
    assert item.image_paths.satellite_path is None
    assert item.sources.website is None


def test_no_resources(setup):
    setup([raw_record()])
    item = MonetaDataset(data_resources=[]).data[0]
    assert item.image_paths is None
    assert item.sources is None


def test_limit(setup):
    setup([{**raw_record(), "id": i} for i in range(5)])
    assert len(MonetaDataset(data_resources=[], limit=2)) == 2


def test_unknown_resource_raises(setup):
    setup([raw_record()])
    with pytest.raises(ValueError, match="Unknown data resources"):
        MonetaDataset(data_resources=["sources"])


def test_missing_images_raise(setup):
    setup([raw_record()], images=False)
    with pytest.raises(FileNotFoundError, match="image"):
        MonetaDataset(data_resources=["satellite"])


def test_source_on_unenriched_data_raises(setup):
    # Raw MONETA.json records hold links in tags, not fetched text
    setup([raw_record(website="https://example.com")])
    with pytest.raises(ValueError, match="not enriched"):
        MonetaDataset(data_resources=["source"])


def test_clues_from_history_and_clues_key(setup):
    from_history = {**raw_record(), "history": {"osm_clues": "osm clue"}}
    from_clues = {
        **raw_record(),
        "id": 2,
        "clues": {"osm": "osm clue", "website": "web clue"},
    }
    setup([from_history, from_clues])
    first, second = MonetaDataset(data_resources=[]).data
    assert first.history.osm_clues == "osm clue"
    assert second.history.osm_clues == "osm clue"
    assert second.history.website_clues == "web clue"


def test_dumped_records_round_trip(setup):
    setup([enriched_record()])
    dumped = [i.model_dump() for i in MonetaDataset(data_resources=["source"]).data]
    setup(dumped)
    item = MonetaDataset(data_resources=["source"]).data[0]
    assert item.name == "Test"
    assert item.sources.wikipedia == "Wikipedia text"


def test_released_moneta_json():
    # Default config: no 'source' -> the released data/Moneta/MONETA.json
    dataset = MonetaDataset(data_resources=[])
    assert len(dataset) == 1000
    assert {item.category for item in dataset.data} == set("ABCDEFGHIJKLMNOPQRSU")
    assert all(item.name for item in dataset.data)


def test_source_without_any_text_raises(setup):
    # Dumped records count as enriched; a run without 'source' leaves them empty
    setup([{**raw_record(), "sources": None}])
    with pytest.raises(ValueError, match="no record"):
        MonetaDataset(data_resources=["source"])

import asyncio

from PIL import Image
import pytest

from moneta.dataloader import moneta as loader
from moneta.preprocess import enrich, map_download, osm_extract, osm_filter, sample
from moneta.utils import load_config, resolve_path

PARAMS = load_config()["preprocess"]


def test_sample_sections_count_and_determinism():
    records = [
        {"id": i, "category": c} for i, c in enumerate("A" * 80 + "B" * 30 + "C" * 60)
    ]
    first = sample.sample_sections(records, n=50, seed=42)
    assert [r["category"] for r in first].count("A") == 50
    assert [r["category"] for r in first].count(
        "B"
    ) == 30  # smaller sections are kept whole
    assert first == sample.sample_sections(records, n=50, seed=42)
    assert [r["category"] for r in first] == sorted(r["category"] for r in first)


def test_gold_requires_name_address_and_link():
    gold = osm_extract.UniqueElementFilter("A", "Gold")
    address = {"name": "X", "addr:street": "S", "addr:housenumber": "1"}
    assert not gold.should_include(address)
    assert gold.should_include({**address, "website": "http://x.de"})
    assert osm_extract.UniqueElementFilter("A", "Bronze").should_include({"name": "X"})


def test_section_tags_from_released_mapping():
    tags = osm_filter.section_tags(resolve_path(load_config()["paths"]["osm_mapping"]))
    assert sorted(tags) == list("ABCDEFGHIJKLMNOPQRSTU")
    assert "landuse=farmland" in tags["A"]


def test_zoom_and_bbox():
    point = map_download.normalize_bbox([10.0, 50.0, 10.0, 50.0], PARAMS)
    assert point[2] - point[0] == pytest.approx(PARAMS["min_span"])
    assert (
        map_download.calculate_best_zoom([10.0, 50.0, 10.0, 50.0], PARAMS)
        == PARAMS["max_zoom"]
    )
    assert (
        map_download.calculate_best_zoom([0.0, 0.0, 10.0, 10.0], PARAMS)
        == PARAMS["min_zoom"]
    )
    assert map_download.is_blank_image(Image.new("RGB", (4, 4)))


def test_apply_fetched_matches_dataloader_format(tmp_path, monkeypatch):
    record = {
        "id": 1,
        "type": "way",
        "bbox": [0, 0, 0, 0],
        "category": "C",
        "tags": {"name": "Mill", "website": "http://mill.de", "wikidata": "Q1"},
    }
    out = enrich.apply_fetched(record, {"website": "We mill flour", "wikidata": ""})
    assert out["tags"]["website"] == "We mill flour"
    assert out["tags"]["wikidata"] == "Q1"  # failed fetch keeps the link
    assert (out["website"], out["wikidata"], out["wikipedia"]) == (
        "http://mill.de",
        "Q1",
        None,
    )
    assert record["tags"]["website"] == "http://mill.de"  # input untouched

    path = tmp_path / "data.json"
    path.write_text(__import__("json").dumps([out]))
    monkeypatch.setattr(
        loader, "load_config", lambda: {"paths": {"data_extended": str(path)}}
    )
    item = loader.MonetaDataset(["source"]).data[0]
    assert item.sources.website == "We mill flour"


def test_enrich_record_keeps_link_on_error():
    async def boom(value, session):
        raise TimeoutError("slow")

    record = {"id": 1, "type": "node", "tags": {"name": "X", "website": "http://x.de"}}
    original = enrich.FETCHERS["website"]
    enrich.FETCHERS["website"] = boom
    try:
        out = asyncio.run(enrich.enrich_record(record, None, asyncio.Semaphore(1)))
    finally:
        enrich.FETCHERS["website"] = original
    assert out["tags"]["website"] == "http://x.de"

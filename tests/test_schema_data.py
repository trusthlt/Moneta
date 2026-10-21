from pydantic import ValidationError
import pytest

from moneta.schema.data import (
    ImagePathsSchema,
    MultimodalInputSchema,
    SourceSchema,
)


def test_image_paths_schema_defaults():
    img = ImagePathsSchema()
    assert img.osm_path is None
    assert img.satellite_path is None


def test_image_paths_schema_values():
    img = ImagePathsSchema(osm_path="osm.png", satellite_path="sat.png")
    assert img.osm_path == "osm.png"
    assert img.satellite_path == "sat.png"


def test_source_schema_defaults():
    src = SourceSchema()
    assert src.wikidata is None
    assert src.wikipedia is None
    assert src.website is None


def test_source_schema_values():
    src = SourceSchema(
        wikidata="Q42", wikipedia="en:Example", website="https://example.com"
    )
    assert src.wikidata == "Q42"
    assert src.wikipedia == "en:Example"
    assert src.website == "https://example.com"


def test_multimodal_input_schema_defaults():
    m = MultimodalInputSchema()
    assert m.id is None
    assert m.type is None
    assert m.name is None
    assert m.bbox is None
    assert m.category is None
    assert m.osm_tags == {}
    assert m.image_paths is None
    assert m.sources is None


def test_multimodal_input_schema_values():
    img = ImagePathsSchema(osm_path="osm.png", satellite_path="sat.png")
    src = SourceSchema(wikidata="Q42")
    m = MultimodalInputSchema(
        id=1,
        type="way",
        name="Test",
        bbox=[1.0, 2.0, 3.0, 4.0],
        category="A",
        osm_tags={"key": "value"},
        image_paths=img,
        sources=src,
    )
    assert m.id == 1
    assert m.type == "way"
    assert m.name == "Test"
    assert m.bbox == [1.0, 2.0, 3.0, 4.0]
    assert m.category == "A"
    assert m.osm_tags == {"key": "value"}
    assert m.image_paths == img
    assert m.sources == src


def test_multimodal_input_schema_invalid_bbox():
    with pytest.raises(ValidationError):
        MultimodalInputSchema(bbox="not_a_list")


def test_multimodal_input_schema_invalid_image_paths():
    with pytest.raises(ValidationError):
        MultimodalInputSchema(image_paths="not_an_imagepaths")

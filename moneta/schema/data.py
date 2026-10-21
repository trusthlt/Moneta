"""
Schemas for multimodal input data.
ImagePathsSchema: Paths to OSM and satellite images.
SourceSchema: Wikidata, Wikipedia, and website content.
HistorySchema: Clues generated from each modality.
MultimodalInputSchema: Combines all above with metadata.
"""

from pydantic import BaseModel


class ImagePathsSchema(BaseModel):
    osm_path: str | None = None
    satellite_path: str | None = None


class SourceSchema(BaseModel):
    wikidata: str | None = None
    wikipedia: str | None = None
    website: str | None = None


class HistorySchema(BaseModel):
    osm_clues: str | None = None
    satellite_clues: str | None = None
    wikidata_clues: str | None = None
    wikipedia_clues: str | None = None
    website_clues: str | None = None


class MultimodalInputSchema(BaseModel):
    id: int | None = None
    type: str | None = None
    name: str | None = None
    bbox: list[float] | None = None
    category: str | None = None
    osm_tags: dict | None = {}
    image_paths: ImagePathsSchema | None = None
    sources: SourceSchema | None = None
    history: HistorySchema | None = None

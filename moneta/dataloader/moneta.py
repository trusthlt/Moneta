"""
Data loader for the MONETA dataset.

Each record becomes a MultimodalInputSchema with id, type, name, bbox, category,
osm_tags and, depending on data_resources:
- osm:       path to the OSM map image
- satellite: path to the satellite image
- source:    fetched Wikidata / Wikipedia / website text (needs the enriched dataset)
Clues (multi-turn) are read from `history.<modality>_clues` or `clues.<modality>`.

Two record formats are accepted:
- raw:    {id, type, tags, bbox, category, ...}; in enriched files the fetched text
          replaces the links in tags.{wikidata,wikipedia,website} (failed fetches keep the link)
- dumped: MultimodalInputSchema dumps (e.g. clue files) with osm_tags / sources / history
"""

import re

from moneta.schema.data import (
    HistorySchema,
    ImagePathsSchema,
    MultimodalInputSchema,
    SourceSchema,
)
from moneta.utils import load_config, read_json_list, resolve_path

VALID_RESOURCES = {"osm", "satellite", "source"}
CLUE_MODALITIES = ["osm", "satellite", "wikidata", "wikipedia", "website"]
SOURCE_KEYS = ["wikidata", "wikipedia", "website"]


def _get_history(elem) -> HistorySchema:
    history = elem.get("history") or {}
    if any(history.values()):
        return HistorySchema(**history)
    clues = elem.get("clues") or {}
    return HistorySchema(**{f"{m}_clues": clues.get(m) for m in CLUE_MODALITIES})


# A raw link as stored in OSM tags: URL, Wikidata QID or Wikipedia "lang:Title"
LINK = re.compile(r"(https?://|www\.)\S+|Q\d+|[a-z-]{2,12}:[^\n]+")


class MonetaDataset:
    def __init__(
        self,
        data_resources: list[str],
        data_path: str | None = None,
        limit: int | None = None,
    ) -> None:
        unknown = set(data_resources) - VALID_RESOURCES
        if unknown:
            raise ValueError(
                f"Unknown data resources {sorted(unknown)}; valid: {sorted(VALID_RESOURCES)}"
            )
        if limit is not None and limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")
        self.config = load_config()
        self.input_types = data_resources
        if data_path is None:
            key = "data_extended" if "source" in data_resources else "data"
            data_path = self.config["paths"][key]
        self.data_path = resolve_path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")

        records = read_json_list(self.data_path)
        if "source" in data_resources:
            self._check_sources(records)
        self.data = [self._modify_input(record) for record in records[:limit]]
        self._check_image_paths()

    def __len__(self) -> int:
        return len(self.data)

    def _modify_input(self, elem):
        tags = elem.get("tags") or elem.get("osm_tags")
        item = {
            "id": elem["id"],
            "type": elem["type"],
            "name": tags.get("name") or elem.get("name"),
            "bbox": elem["bbox"],
            "category": elem["category"],
            "osm_tags": tags,
            "history": _get_history(elem),
        }
        if self.input_types:
            item["image_paths"] = self._get_image_paths(elem)
            item["sources"] = self._get_sources(elem)
        return MultimodalInputSchema(**item)

    def _get_sources(self, elem):
        if "source" not in self.input_types:
            return SourceSchema()
        if "sources" in elem:
            return SourceSchema(**(elem["sources"] or {}))
        tags = elem.get("tags") or {}
        return SourceSchema(**{key: tags.get(key) for key in SOURCE_KEYS})

    def _get_image_paths(self, elem):
        file_name = f"{elem['type']}_{elem['id']}.png"
        paths = {}
        if "osm" in self.input_types:
            paths["osm_path"] = str(
                resolve_path(self.config["paths"]["osm_images"]) / file_name
            )
        if "satellite" in self.input_types:
            paths["satellite_path"] = str(
                resolve_path(self.config["paths"]["satellite_images"]) / file_name
            )
        return ImagePathsSchema(**paths)

    def _check_sources(self, records):
        """Fail if the file is not enriched: (almost) all source values are still links.
        Enriched paper files: ~25% links (failed fetches); raw MONETA.json: >99%."""
        values = [
            v.strip()
            for r in records
            for v in (
                (r.get("sources") or {}).values()
                if "sources" in r
                else [r["tags"].get(k) for k in SOURCE_KEYS]
            )
            if v
        ]
        links = sum(bool(LINK.fullmatch(v)) for v in values)
        if not values or links / len(values) > 0.9:
            raise ValueError(
                f"'source' requested but {self.data_path} has no fetched source text "
                f"({links}/{len(values)} values are links). Use the enriched dataset."
            )

    def _check_image_paths(self):
        missing = [
            path
            for item in self.data
            if item.image_paths
            for path in (item.image_paths.osm_path, item.image_paths.satellite_path)
            if path and not resolve_path(path).exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} image(s) missing, e.g. {missing[0]}. Download the images first."
            )

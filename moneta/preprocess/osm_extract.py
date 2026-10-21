"""
Step 2: collect the filtered objects of all sections into one file.
Bronze: name; Silver: + street and house number; Gold (MONETA): + Wikidata, Wikipedia or website link.
Reads <paths.processed>/nace/nace_<section>.osm.pbf, writes <paths.processed>/nace_combined_<mode>.json
"""

import argparse
from pathlib import Path

import osmium

from moneta.utils import get_logger, load_config, resolve_path, write_json

logger = get_logger()


class UniqueElementFilter(osmium.SimpleHandler):
    def __init__(self, category, include_mode):
        super().__init__()
        self.elements = []
        self.node_locations = {}
        self.include_mode = include_mode
        self.category = category

    def should_include(self, tags):
        if self.include_mode == "Bronze":
            return "name" in tags
        elif self.include_mode == "Silver":
            return (
                "name" in tags and "addr:housenumber" in tags and "addr:street" in tags
            )
        return (
            "name" in tags
            and "addr:housenumber" in tags
            and "addr:street" in tags
            and any(k in tags for k in ["wikidata", "wikipedia", "website"])
        )

    def node(self, n):
        if not n.location.valid():
            return
        self.node_locations[n.id] = (n.location.lon, n.location.lat)
        if self.should_include(n.tags):
            bbox = [n.location.lon, n.location.lat, n.location.lon, n.location.lat]
            self.elements.append(self.format_element(n, bbox))

    def way(self, w):
        coords = [
            self.node_locations.get(n.ref)
            for n in w.nodes
            if n.ref in self.node_locations
        ]
        coords = [c for c in coords if c]
        if self.should_include(w.tags) and coords:
            bbox = self.compute_bbox(coords)
            self.elements.append(self.format_element(w, bbox))

    def relation(self, r):
        coords = []
        for m in r.members:
            if m.type == "n" and m.ref in self.node_locations:
                coords.append(self.node_locations[m.ref])
        if self.should_include(r.tags) and coords:
            bbox = self.compute_bbox(coords)
            self.elements.append(self.format_element(r, bbox))

    def compute_bbox(self, coords):
        lons, lats = zip(*coords, strict=False)
        return [min(lons), min(lats), max(lons), max(lats)]

    def format_element(self, elem, bbox):
        return {
            "id": elem.id,
            "type": elem.__class__.__name__.lower(),
            "tags": dict(elem.tags),
            "bbox": bbox,
            "category": self.category,
        }


def main(mode: str = "Gold") -> Path:
    processed = resolve_path(load_config()["paths"]["processed"])
    elements = []
    for path in sorted((processed / "nace").glob("nace_*.osm.pbf")):
        section = path.name.removeprefix("nace_").split(".")[0]
        handler = UniqueElementFilter(section, mode)
        handler.apply_file(str(path))
        elements.extend(handler.elements)
        logger.info(f"{path.name}: {len(handler.elements)} elements")
    if not elements:
        raise FileNotFoundError(
            f"No filtered elements in {processed / 'nace'}; run osm_filter first"
        )
    out = processed / f"nace_combined_{mode}.json"
    write_json(elements, out)
    logger.info(f"Wrote {len(elements)} elements to {out}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="Gold", choices=["Bronze", "Silver", "Gold"])
    main(parser.parse_args().mode)

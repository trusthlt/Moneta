"""
Step 1: per NACE section, keep the OSM objects (and the objects they reference) matching the
section's tags in data/filter/osm_nace_mapping.xlsx. Needs the osmium-tool CLI.
Writes <paths.processed>/nace/nace_<section>.osm.pbf
"""

import argparse
import ast
from pathlib import Path
import shutil
import subprocess

import pandas as pd

from moneta.utils import get_logger, load_config, resolve_path

logger = get_logger()


def section_tags(mapping_path) -> dict[str, list[str]]:
    df = pd.read_excel(mapping_path)
    return {
        row["Category"]: ast.literal_eval(row["OSM Tags"]) for _, row in df.iterrows()
    }


def main(input_pbf, sections=None) -> Path:
    if shutil.which("osmium") is None:
        raise RuntimeError(
            "osmium-tool CLI not found, e.g. `conda install -c conda-forge osmium-tool`"
        )
    paths = load_config()["paths"]
    out_dir = resolve_path(paths["processed"]) / "nace"
    out_dir.mkdir(parents=True, exist_ok=True)
    for section, tags in section_tags(resolve_path(paths["osm_mapping"])).items():
        if sections and section not in sections:
            continue
        if not tags:
            logger.warning(f"Section {section}: no OSM tags, skipped")
            continue
        out = out_dir / f"nace_{section}.osm.pbf"
        cmd = [
            "osmium",
            "tags-filter",
            str(input_pbf),
            *[f"nwr/{t}" for t in tags],
            "-o",
            str(out),
            "--overwrite",
        ]
        logger.info(" ".join(cmd))
        subprocess.run(cmd, check=True)
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="OSM extract, e.g. europe-latest.osm.pbf (Geofabrik)",
    )
    parser.add_argument(
        "--sections", nargs="+", default=None, help="Default: all sections"
    )
    args = parser.parse_args()
    main(args.input, args.sections)

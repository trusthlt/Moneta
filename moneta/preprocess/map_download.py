"""
Step 5: OSM map and ESRI satellite images per record: tiles covering the (padded) bounding box
at the largest zoom with <= max_tiles tiles, stitched; zoom is lowered while a tile set is blank.
Writes <out>/osm/<type>_<id>.png and <out>/satellite/<type>_<id>.png (existing images are skipped).
"""

import argparse
import asyncio
from io import BytesIO
import math

import aiohttp
import mercantile
from PIL import Image
from tqdm.asyncio import tqdm

from moneta.utils import get_logger, load_config, read_json_list, resolve_path

logger = get_logger()


def is_blank_image(img: Image.Image) -> bool:
    """Single-colour image (e.g. empty or missing tiles)."""
    return len(img.getcolors(img.size[0] * img.size[1])) == 1


def calculate_best_zoom(bbox, params) -> int:
    max_deg = max(abs(bbox[2] - bbox[0]), abs(bbox[3] - bbox[1]))
    if max_deg == 0:
        return params["max_zoom"]
    zoom = math.ceil(math.log2(360 * params["max_tiles"] / max_deg))
    return min(max(zoom, params["min_zoom"]), params["max_zoom"])


def normalize_bbox(bbox, params):
    """Pad spans smaller than min_span (e.g. single nodes)."""
    minlon, minlat, maxlon, maxlat = bbox
    span = params["min_span"]
    if abs(maxlat - minlat) < span:
        minlat, maxlat = minlat - span / 2, maxlat + span / 2
    if abs(maxlon - minlon) < span:
        minlon, maxlon = minlon - span / 2, maxlon + span / 2
    return [minlon, minlat, maxlon, maxlat]


def tile_grid(bbox, zoom):
    tiles = list(mercantile.tiles(*bbox, zoom))
    if not tiles:
        raise ValueError(f"No tiles for {bbox} at zoom {zoom}")
    return sorted({t.x for t in tiles}), sorted({t.y for t in tiles})


async def download_tile(url_template, z, x, y, session, semaphore):
    url = url_template.format(z=z, x=x, y=y)
    async with semaphore:
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    return Image.open(BytesIO(await response.read()))
                logger.warning(f"Tile {url}: HTTP {response.status}")
        except (aiohttp.ClientError, TimeoutError) as e:
            logger.warning(f"Tile {url}: {e}")
    return None


async def stitched_image(
    url_template, zoom, x_tiles, y_tiles, session, semaphore, tile_size
):
    """Stitched image, or None if any tile failed to download."""
    img = Image.new("RGB", (len(x_tiles) * tile_size, len(y_tiles) * tile_size))
    for row, y in enumerate(y_tiles):
        tiles = await asyncio.gather(
            *(
                download_tile(url_template, zoom, x, y, session, semaphore)
                for x in x_tiles
            )
        )
        if any(tile is None for tile in tiles):
            return None
        for col, tile in enumerate(tiles):
            img.paste(tile, (col * tile_size, row * tile_size))
    return img


async def save_record_images(record, out_dirs, params, session, semaphores):
    name = f"{record['type']}_{record['id']}.png"
    if (out_dirs["osm"] / name).exists() and (out_dirs["satellite"] / name).exists():
        return
    async with semaphores["record"]:
        bbox = normalize_bbox(record["bbox"], params)
        zoom = calculate_best_zoom(bbox, params)
        while True:
            x_tiles, y_tiles = tile_grid(bbox, zoom)
            osm, sat = await asyncio.gather(
                *(
                    stitched_image(
                        params[f"{k}_tiles"],
                        zoom,
                        x_tiles,
                        y_tiles,
                        session,
                        semaphores["tile"],
                        params["tile_size"],
                    )
                    for k in ("osm", "satellite")
                )
            )
            if osm is None or sat is None:
                logger.warning(
                    f"{name}: tile download failed, not saved (rerun to retry)"
                )
                return
            if (
                not (is_blank_image(osm) or is_blank_image(sat))
                or zoom <= params["min_zoom"]
            ):
                break
            zoom -= 1
        osm.save(out_dirs["osm"] / name)
        sat.save(out_dirs["satellite"] / name)


async def download(records, out_dirs, params):
    semaphores = {"record": asyncio.Semaphore(3), "tile": asyncio.Semaphore(5)}
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(
        headers={"User-Agent": params["user_agent"]}, timeout=timeout
    ) as session:
        await tqdm.gather(
            *(
                save_record_images(r, out_dirs, params, session, semaphores)
                for r in records
            ),
            desc="Records",
        )


def main(input_path=None, output_dir=None, limit=None) -> dict:
    config = load_config()
    records = read_json_list(resolve_path(input_path or config["paths"]["data"]))[
        :limit
    ]
    if output_dir:
        out_dirs = {k: resolve_path(output_dir) / k for k in ("osm", "satellite")}
    else:
        out_dirs = {
            k: resolve_path(config["paths"][f"{k}_images"])
            for k in ("osm", "satellite")
        }
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    asyncio.run(download(records, out_dirs, config["preprocess"]))
    logger.info(f"Images for {len(records)} records in {out_dirs['osm'].parent}")
    return out_dirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=None,
        help="Default: paths.data (bbox from the released MONETA.json)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Default: paths.osm_images / paths.satellite_images",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    main(args.input, args.output_dir, args.limit)

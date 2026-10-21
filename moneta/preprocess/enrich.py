"""
Step 4: fetch Wikidata / Wikipedia / website text for each record's links.
Output (read by MonetaDataset): the fetched text replaces the link in tags.<source>; a failed
fetch keeps the link (as in the paper files) and is logged. The original links are kept top-level.
Unlike the paper's 1K file, no geocoded addr:* fields are added (not used by the experiments).
Fetched web content changes over time, so a new run will not match the paper's text exactly.
"""

import argparse
import asyncio
import copy
import json
from pathlib import Path

import aiohttp
from bs4 import BeautifulSoup
from tqdm.asyncio import tqdm

from moneta.utils import (
    get_logger,
    load_config,
    read_json_list,
    resolve_path,
    write_json,
)

logger = get_logger()
SOURCE_KEYS = ["wikidata", "wikipedia", "website"]


async def get(session, url, params=None, as_json=True, retries=3):
    """GET with retries on 429/5xx; raises on other errors (logged by enrich_record)."""
    for attempt in range(retries):
        async with session.get(url, params=params) as response:
            if response.status == 429 or response.status >= 500:
                await asyncio.sleep(2**attempt)
                continue
            response.raise_for_status()
            return (
                await response.json(content_type=None)
                if as_json
                else await response.text()
            )
    raise aiohttp.ClientError(f"{url}: still failing after {retries} attempts")


async def fetch_wikipedia(wikipedia_tag: str, session) -> str:
    """'lang:Title' -> 'Title: <title>\\n<plain-text extract>'."""
    try:
        lang, title = wikipedia_tag.split(":", 1)
    except ValueError:
        return ""
    title = title.replace(" ", "_")
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "titles": title,
        "explaintext": "1",
        "redirects": "1",
    }
    data = await get(session, f"https://{lang}.wikipedia.org/w/api.php", params)
    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return ""
    return f"Title: {title}\n{next(iter(pages.values())).get('extract', '')}"


async def fetch_labels(ids: list[str], session, language="en") -> dict:
    """Labels for Wikidata Q/P ids; the API takes at most 50 ids per request (unresolved ids keep their id)."""
    labels = {}
    for start in range(0, len(ids), 50):
        params = {
            "action": "wbgetentities",
            "ids": "|".join(ids[start : start + 50]),
            "format": "json",
            "languages": language,
            "props": "labels",
        }
        try:
            data = await get(session, "https://www.wikidata.org/w/api.php", params)
        except aiohttp.ClientError as e:
            logger.warning(f"Wikidata labels: {e}")
            continue
        for eid, ent in data.get("entities", {}).items():
            labels[eid] = ent.get("labels", {}).get(language, {}).get("value", eid)
    return labels


async def fetch_wikidata(qid: str, session, language="en") -> str:
    """Entity label + claims with resolved property / value labels, as indented JSON."""
    data = await get(
        session, f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
    )
    entity = next(iter(data["entities"].values()))
    claims = entity.get("claims", {})
    value_ids = [
        s.get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("id")
        for statements in claims.values()
        for s in statements
        if s.get("mainsnak", {}).get("datavalue", {}).get("type") == "wikibase-entityid"
    ]
    labels = await fetch_labels(sorted(set(claims) | set(value_ids)), session, language)
    parsed = {
        "label": entity.get("labels", {})
        .get(language, {})
        .get("value", "Unknown entity"),
        "claims": {},
    }
    for prop_id, statements in claims.items():
        values = []
        for statement in statements:
            dv = statement.get("mainsnak", {}).get("datavalue", {})
            if dv.get("type") == "wikibase-entityid":
                values.append(labels.get(dv["value"]["id"], dv["value"]["id"]))
            elif isinstance(dv.get("value"), dict) and "time" in dv["value"]:
                values.append(dv["value"]["time"].lstrip("+"))
            elif isinstance(dv.get("value"), str):
                values.append(dv["value"])
        if values:
            parsed["claims"][labels.get(prop_id, prop_id)] = values
    return json.dumps(parsed, indent=2)


async def fetch_website(url: str, session) -> str:
    html = await get(session, url, as_json=False, retries=1)
    return BeautifulSoup(html, "html.parser").get_text(separator=" ", strip=True)


FETCHERS = {
    "wikidata": fetch_wikidata,
    "wikipedia": fetch_wikipedia,
    "website": fetch_website,
}


def apply_fetched(record: dict, fetched: dict) -> dict:
    """Original links top-level; non-empty fetched text replaces the link in tags."""
    out = copy.deepcopy(record)
    for key in SOURCE_KEYS:
        out[key] = record["tags"].get(key)
        text = fetched.get(key)
        if isinstance(text, str) and text.strip():
            out["tags"][key] = text
    return out


async def enrich_record(record, session, semaphore):
    links = {k: record["tags"][k] for k in SOURCE_KEYS if record["tags"].get(k)}
    async with semaphore:
        results = await asyncio.gather(
            *(FETCHERS[k](v, session) for k, v in links.items()), return_exceptions=True
        )
    fetched = {}
    for key, result in zip(links, results, strict=True):
        if isinstance(result, Exception):
            logger.warning(
                f"{record['type']}_{record['id']} {key}: {type(result).__name__}: {result}"
            )
        else:
            fetched[key] = result
    return apply_fetched(record, fetched)


async def enrich(records, user_agent: str, concurrency: int = 2):
    timeout = aiohttp.ClientTimeout(total=60)
    semaphore = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession(
        headers={"User-Agent": user_agent}, timeout=timeout
    ) as session:
        return await tqdm.gather(
            *(enrich_record(r, session, semaphore) for r in records), desc="Records"
        )


def main(input_path=None, output_path=None, limit=None, concurrency=2) -> Path:
    config = load_config()
    input_path = resolve_path(input_path or config["paths"]["data"])
    output_path = resolve_path(
        output_path or Path(config["paths"]["processed"]) / "MONETA_extended_new.json"
    )
    records = read_json_list(input_path)[:limit]
    enriched = asyncio.run(
        enrich(records, config["preprocess"]["user_agent"], concurrency)
    )
    n_text = sum(
        enriched[i]["tags"].get(k) != records[i]["tags"].get(k)
        for i in range(len(records))
        for k in SOURCE_KEYS
    )
    n_links = sum(bool(r["tags"].get(k)) for r in records for k in SOURCE_KEYS)
    write_json(enriched, output_path)
    logger.info(
        f"Fetched text for {n_text}/{n_links} links; wrote {len(enriched)} records to {output_path}"
    )
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default=None, help="Default: paths.data (data/Moneta/MONETA.json)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Default: <paths.processed>/MONETA_extended_new.json",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args()
    main(args.input, args.output, args.limit, args.concurrency)

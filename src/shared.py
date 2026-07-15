import json
import unicodedata
from pathlib import Path

import requests

OKOBAU_URL = "https://oekobaudat.de/OEKOBAU.DAT/resource/datastocks/c391de0f-2cfd-47ea-8883-c661d294e2ba"
REQUEST_TIMEOUT = 30


def get_epds(limit=10, start_index=0) -> dict:
    """Get EPDs from Ökobau."""

    response = requests.get(
        f"{OKOBAU_URL}/processes?format=json&pageSize={limit}&startIndex={start_index}",
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()

    print(f"Retrieved {data.get('pageSize')} EPDs out of {data.get('totalCount')} from Ökobau")

    return data


def get_all_epds() -> list[dict]:
    """Get all EPD metadata in one request."""

    data = get_epds(limit=3000)
    return data.get("data", [])


def get_full_epd(uid: str) -> dict:
    """Get the full dataset for a single EPD."""

    base_url = f"{OKOBAU_URL}/processes/{uid}"
    response = requests.get(f"{base_url}?format=json&view=extended", timeout=REQUEST_TIMEOUT)

    response.raise_for_status()
    data = response.json()
    data["source"] = base_url

    return data


def get_full_epd_str(uid: str) -> str:
    """Get the full dataset for a single EPD and return it as a string."""
    return json.dumps(get_full_epd(uid))


def get_folder(source, name: str) -> Path:
    folder = Path(source).parent / name
    if not folder.exists():
        folder.mkdir()

    return folder


def normalize_text(value: str) -> str:
    """Normalize text for case-insensitive comparisons."""

    normalized = unicodedata.normalize("NFKC", value or "")
    return " ".join(normalized.casefold().split())


def find_epds_by_name(object_name: str, exact=False) -> list[dict]:
    """Find EPD metadata entries by object name."""

    needle = normalize_text(object_name)
    matches = []
    for epd in get_all_epds():
        epd_name = epd.get("name", "")
        haystack = normalize_text(epd_name)
        is_match = haystack == needle if exact else needle in haystack
        if is_match:
            matches.append(epd)

    return matches


def extract_epd_name(dataset: dict, preferred_lang="en") -> str | None:
    """Extract the dataset name from the ILCD structure."""

    names = (
        dataset.get("processInformation", {})
        .get("dataSetInformation", {})
        .get("name", {})
        .get("baseName", [])
    )
    for entry in names:
        if entry.get("lang") == preferred_lang and entry.get("value"):
            return entry["value"]
    for entry in names:
        if entry.get("value"):
            return entry["value"]
    return None


def get_lca(dataset: dict) -> list[dict]:
    """Extract the LCA results embedded in an EPD dataset."""

    lcia_results = dataset.get("LCIAResults", {}).get("LCIAResult", [])
    if not lcia_results:
        uid = (
            dataset.get("processInformation", {})
            .get("dataSetInformation", {})
            .get("UUID")
        )
        raise ValueError(f"No LCA data found for EPD {uid}")

    results = []
    for result in lcia_results:
        descriptions = result.get("referenceToLCIAMethodDataSet", {}).get("shortDescription", [])
        indicator = next(
            (
                item.get("value")
                for item in descriptions
                if item.get("lang") == "en" and item.get("value")
            ),
            None,
        )
        if not indicator:
            indicator = next((item.get("value") for item in descriptions if item.get("value")), None)

        modules = []
        unit = None

        other_data = result.get("other", {})

        print(
            "Chiavi disponibili dentro LCIAResult.other:",
            list(other_data.keys()),
        )

        for entry in other_data.get("anies", []):
            if entry.get("name") == "referenceToUnitGroupDataSet":
                short_descriptions = (
                    entry.get("value", {})
                    .get("shortDescription", [])
                )

                unit = next(
                    (
                        item.get("value")
                        for item in short_descriptions
                        if item.get("value")
                    ),
                    None,
                )

                continue

            if "module" in entry:

                raw_value = entry.get("value")
                modules.append(
                    {
                        "module": entry.get("module"),
                        "value": "0.0" if raw_value is None else raw_value,
                        "scenario": entry.get("scenario"),
                    }
                )

        results.append(
            {
                "indicator": indicator,
                "unit": unit,
                "modules": modules,
            }
        )

    return results


def get_lca_by_object_name(object_name: str, exact=False) -> dict:
    """Find the first matching EPD by name and return its LCA results."""

    matches = find_epds_by_name(object_name, exact=exact)
    if not matches:
        raise ValueError(f"No EPD found for object name '{object_name}'")

    if len(matches) > 1:
        raise ValueError(
            f"Multiple EPDs found for '{object_name}'. "
            "Use a more specific name or inspect the returned matches first."
        )

    return get_lca_by_epd_match(matches[0], object_name=object_name)


def get_lca_by_epd_match(matched_epd: dict, object_name: str | None = None) -> dict:
    """Get LCA results starting from an already matched EPD metadata entry."""

    dataset = get_full_epd(matched_epd["uuid"])
    return {
        "query": object_name or matched_epd.get("name"),
        "match": {
            "uuid": matched_epd.get("uuid"),
            "name": matched_epd.get("name"),
            "dataset_name": extract_epd_name(dataset),
        },
        "lca": get_lca(dataset),
    }

def get_epd_names(language='de') -> list[dict]:
    """Get UUIDs and names of all EPDs"""
    edps = get_all_epds()
    results = []
    for epd in edps:
        uuid = epd.get("uuid")
        name = epd.get("name")

        if not uuid:
            continue
        results.append({
            "uuid": uuid,
            "name": name,
            "language": language
        })
    return results

def get_all_epd_names() -> list[dict]:
    """Get UUIDs and names of all EPDs in all languages"""
    epds = get_all_epds()
    return [
        {
            "uuid": epd.get("uuid"),
            "name": epd.get("name"),
        }
        for epd in epds
        if epd.get("name")
    ]

def get_lca_by_uuid(uuid: str, name: str | None = None) -> dict:
    """Get complete LCA dataset for a given EPD with UUID""" 
    dataset = get_full_epd(uuid)

    return {
        "uuid": uuid,
        "name": name or extract_epd_name(dataset),
        "dataset_name": extract_epd_name(dataset),
        "source": dataset.get("source"),
        "lca": get_lca(dataset),
    }
"""Look up NYC HPD housing-code violations for one building (NYC Open Data, dataset wvxf-dwi5).

Usage:  python hpd_lookup.py "220 W 26th St" MANHATTAN
Prints JSON: matched address, open-violation counts by class, and the most recent open violations.
Standard library only (runs in the managed-agent sandbox and locally as the fallback).
"""
from __future__ import annotations

import json
import re
import sys
import urllib.parse
import urllib.request

API = "https://data.cityofnewyork.us/resource/wvxf-dwi5.json"
BOROUGHS = {"MANHATTAN", "BRONX", "BROOKLYN", "QUEENS", "STATEN ISLAND"}
DIRS = {"W": "WEST", "E": "EAST", "N": "NORTH", "S": "SOUTH"}
SUFFIXES = {"ST": "STREET", "AVE": "AVENUE", "AV": "AVENUE", "PL": "PLACE", "RD": "ROAD", "BLVD": "BOULEVARD",
            "PKWY": "PARKWAY", "DR": "DRIVE", "LN": "LANE", "CT": "COURT", "TER": "TERRACE", "SQ": "SQUARE"}


def normalize(address: str) -> tuple[str, str]:
    """'220 W 26th St, Apt 4B' -> ('220', 'WEST 26 STREET') in HPD's street-name style."""
    first = address.split(",")[0].upper().strip()
    m = re.match(r"^\s*([\d-]+[A-Z]?)\s+(.+)$", first)
    if not m:
        raise ValueError(f"Can't find a house number in {address!r}")
    number, street = m.group(1), m.group(2)
    words = []
    for w in re.split(r"\s+", street.replace(".", "")):
        w = re.sub(r"^(\d+)(ST|ND|RD|TH)$", r"\1", w)          # 26TH -> 26
        w = DIRS.get(w, w) if not words else w                  # leading W -> WEST
        w = SUFFIXES.get(w, w)
        words.append(w)
    return number, " ".join(words)


def get(params: dict) -> list[dict]:
    """GET the Socrata API with query params; return parsed JSON rows."""
    url = API + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def lookup(address: str, borough: str, limit: int = 15) -> dict:
    """Return {"matched", "total", "open_total", "open_by_class", "recent_open": [...]} for the building."""
    borough = borough.upper().strip()
    if borough not in BOROUGHS:
        raise ValueError(f"Unknown borough {borough!r}")
    number, street = normalize(address)
    where = f"housenumber='{number}' AND boro='{borough}' AND upper(streetname)='{street}'"
    counts = get({"$select": "violationstatus, class, count(*) as n", "$where": where,
                  "$group": "violationstatus, class"})
    open_rows = get({"$select": "violationid, class, novdescription, inspectiondate, apartment, story",
                     "$where": where + " AND violationstatus='Open'", "$order": "inspectiondate DESC",
                     "$limit": limit})
    open_by_class = {c["class"]: int(c["n"]) for c in counts if c["violationstatus"] == "Open"}
    return {
        "matched": {"housenumber": number, "streetname": street, "boro": borough},
        "total": sum(int(c["n"]) for c in counts),
        "open_total": sum(open_by_class.values()),
        "open_by_class": open_by_class,
        "recent_open": [{"violationid": r["violationid"], "class": r.get("class"),
                         "inspectiondate": (r.get("inspectiondate") or "")[:10], "apartment": r.get("apartment"),
                         "description": re.sub(r"^.*?ADM CODE\s*", "", r.get("novdescription", ""))[:220]}
                        for r in open_rows],
        "source": "NYC Open Data, HPD Housing Maintenance Code Violations (wvxf-dwi5)",
    }


if __name__ == "__main__":
    print(json.dumps(lookup(sys.argv[1], sys.argv[2]), indent=1))

---
name: hpd-lookup
description: Look up NYC HPD housing-code violations (open and closed) for one building address on NYC Open Data.
---
# HPD violation lookup

Use this when you need a building's HPD violation history.

Run:

```
python .agents/skills/hpd-lookup/hpd_lookup.py "<house number and street>" <BOROUGH>
```

Example: `python .agents/skills/hpd-lookup/hpd_lookup.py "220 W 26th St" MANHATTAN`

- BOROUGH is one of MANHATTAN, BRONX, BROOKLYN, QUEENS, "STATEN ISLAND".
- The script normalizes the street to HPD's style (e.g. "W 26th St" -> "WEST 26 STREET") and prints JSON:
  `matched`, `total`, `open_total`, `open_by_class` (A = non-hazardous, B = hazardous,
  C = immediately hazardous), and `recent_open` (most recent open violations with descriptions).
- If it returns `total: 0`, the street name may be spelled differently in HPD data; try one variant
  (e.g. "AVENUE" vs "AVE", or the full street name) before concluding there are no records.
- Data source: https://data.cityofnewyork.us/resource/wvxf-dwi5.json (public, no key needed).

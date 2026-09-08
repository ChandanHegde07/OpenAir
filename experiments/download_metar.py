"""Download 2025 METAR/ASOS for the 10 challenge airports from Iowa Mesonet.

Public observational weather. Cached under data/external/metar/.
"""
from __future__ import annotations

import urllib.parse
import urllib.request
from pathlib import Path

from common import AIRPORTS, DATA

OUT = DATA / "external" / "metar"
BASE = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"


def url_for(station: str) -> str:
    q = [
        ("station", station),
        ("data", "tmpf"),
        ("data", "dwpf"),
        ("data", "relh"),
        ("data", "drct"),
        ("data", "sknt"),
        ("data", "p01i"),
        ("data", "vsby"),
        ("data", "gust"),
        ("data", "skyc1"),
        ("data", "wxcodes"),
        ("year1", "2025"),
        ("month1", "1"),
        ("day1", "1"),
        ("year2", "2026"),
        ("month2", "1"),
        ("day2", "1"),
        ("tz", "UTC"),
        ("format", "onlycomma"),
        ("latlon", "no"),
        ("elev", "no"),
        ("missing", "null"),
        ("trace", "0.0001"),
        ("direct", "no"),
        ("report_type", "3"),
    ]
    return BASE + "?" + urllib.parse.urlencode(q)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for st in AIRPORTS:
        dest = OUT / f"{st}.csv"
        if dest.exists() and dest.stat().st_size > 1000:
            print(f"skip {st} ({dest.stat().st_size} bytes)", flush=True)
            continue
        u = url_for(st)
        print(f"GET {st} ...", flush=True)
        req = urllib.request.Request(u, headers={"User-Agent": "OpenAir-PRC2026/1.0 (research)"})
        with urllib.request.urlopen(req, timeout=180) as resp:
            body = resp.read()
        dest.write_bytes(body)
        nlines = body.count(b"\n")
        print(f"  wrote {dest} lines={nlines} bytes={len(body)}", flush=True)
        if nlines < 10:
            print(f"  WARN few rows for {st}", flush=True)


if __name__ == "__main__":
    main()

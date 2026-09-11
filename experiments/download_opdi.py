"""Download PRC/OSN OPDI open parquet (flight list + optional events).

Source: https://www.opdi.aero/  (EUROCONTROL PRC + OpenSky)
URL pattern documented on the OPDI download pages.
SSL verify is disabled because eurocontrol.int currently serves an expired cert.
"""
from __future__ import annotations

import ssl
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "external" / "opdi"
CTX = ssl._create_unverified_context()
BASE = "https://www.eurocontrol.int/performance/data/download/OPDI/v002"


def fetch(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"  exists {dest.name} ({dest.stat().st_size:,} bytes)", flush=True)
        return dest
    print(f"  GET {url}", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "OpenAir-PRC2026-research"})
    with urllib.request.urlopen(req, timeout=600, context=CTX) as r, tmp.open("wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(dest)
    print(f"  wrote {dest} ({dest.stat().st_size:,} bytes)", flush=True)
    return dest


def flight_list_url(yyyymm: str) -> str:
    return f"{BASE}/flight_list/flight_list_{yyyymm}.parquet"


def flight_list_months(year: int = 2025) -> list[str]:
    return [f"{year}{m:02d}" for m in range(1, 13)]

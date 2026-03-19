import requests
import pandas as pd
import time
import xarray as xr
from pathlib import Path
from datetime import datetime
sys.path.insert(0, "..")
import CT, FN

# ── Configuration ────────────────────────────────────────────────────────────
API_KEY = 'WGIBI4lbwMEiGOnsCZY7gIVcWweUlVueMjvqE1cB'

FACILITIES = {
    'Alcoa_Allowance_Management_Inc': 6705,
    'Colstrip':                        6076,
    'Gen_J_M_Gavin':                   8102,
    'Hunter':                          6165,
    'Intermountain':                   6481,
    'James_H_Miller_Jr':               6002,
    'Labadie':                         2103,
    'Martin_Lake':                     6146,
    'Miami_Fort_Power_Station':        2832,
    'Milton_R_Young':                  2823,
    'New_Madrid_Power_Plant':          2167,
    'Thomas_Hill_Energy_Center':       2168,
}

YEARS      = list(range(2019, 2026))
OUTPUT_DIR = Path(CT.d_cems)
LOG_FILE   = Path('facility_failures.log')
URL        = 'https://api.epa.gov/easey/streaming-services/facilities/attributes'

DELAY       = 1.0
RETRY_WAIT  = 60
MAX_RETRIES = 5
# ─────────────────────────────────────────────────────────────────────────────


def log_failure(facility_id, name, reason):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(LOG_FILE, 'a') as f:
        f.write(f"{timestamp} | id={facility_id} | {name} | {reason}\n")


def download_facility(facility_id):
    params = {
        'api_key':    API_KEY,
        'facilityId': facility_id,
        'year':       '|'.join(str(y) for y in YEARS),
    }

    for attempt in range(1, MAX_RETRIES + 1):
        resp = requests.get(URL, params=params, timeout=120)

        if resp.status_code == 200:
            data = resp.json()
            return resp.status_code, pd.DataFrame(data) if data else None

        elif resp.status_code == 429:
            print(f"429 rate limit (attempt {attempt}/{MAX_RETRIES}), waiting {RETRY_WAIT}s ...", flush=True)
            time.sleep(RETRY_WAIT)

        else:
            return resp.status_code, None

    return 429, None


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    for name, fid in FACILITIES.items():
        out_path = OUTPUT_DIR / f"{fid}_{name}.csv"

        if out_path.exists():
            print(f"{fid}_{name} — already exists, skipping")
            continue

        print(f"{fid}_{name} ...", end=' ', flush=True)
        status, df = download_facility(fid)

        if df is not None and not df.empty:
            df.to_csv(out_path, index=False)
            print(f"saved {len(df)} rows -> {out_path}")
        elif df is not None:
            print("no data returned")
            log_failure(fid, name, f"HTTP {status} — no data returned")
        else:
            reason = f"HTTP {status} — max retries exceeded" if status == 429 else f"HTTP {status}"
            print(f"failed ({reason})")
            log_failure(fid, name, reason)

        time.sleep(DELAY)

    print("\nDone.")


main()

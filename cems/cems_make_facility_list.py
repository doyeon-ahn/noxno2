import requests
import pandas as pd
import time
sys.path.insert(0, "..")
import CT, FN

# ── Configuration ────────────────────────────────────────────────────────────
API_KEY = 'WGIBI4lbwMEiGOnsCZY7gIVcWweUlVueMjvqE1cB'
YEARS   = list(range(2019, 2026))
URL     = 'https://api.epa.gov/easey/streaming-services/facilities/attributes'
# ─────────────────────────────────────────────────────────────────────────────


def download_all_facilities(year):
    params = {
        'api_key': API_KEY,
        'year':    year,
    }
    resp = requests.get(URL, params=params, timeout=120)
    if resp.status_code == 200:
        return pd.DataFrame(resp.json())
    else:
        print(f"  HTTP {resp.status_code} for year {year} — skipping")
        return None


def main():
    all_dfs = []

    for year in YEARS:
        print(f"Downloading year {year} ...", end=' ', flush=True)
        df = download_all_facilities(year)
        if df is not None and not df.empty:
            print(f"{len(df):,} rows")
            all_dfs.append(df)
        else:
            print("no data")
        time.sleep(1.0)

    # Combine all years
    combined = pd.concat(all_dfs, ignore_index=True)

    # Keep only key columns
    combined = combined[['facilityId', 'facilityName', 'stateCode',
                          'latitude', 'longitude', 'primaryFuelInfo']]

    # Rename for clarity
    combined = combined.rename(columns={
        'stateCode':       'state',
        'primaryFuelInfo': 'primaryFuel',
    })

    # One row per facility: drop duplicates, keeping the most recent entry
    combined = combined.drop_duplicates(subset='facilityId', keep='last')
    combined = combined.sort_values('facilityId').reset_index(drop=True)

    combined.to_csv('cems_list.csv', index=False)
    print(f"\nSaved {len(combined)} facilities -> cems_list.csv")


main()

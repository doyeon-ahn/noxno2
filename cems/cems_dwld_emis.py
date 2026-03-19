import requests, time
import pandas as pd
from pathlib import Path
from datetime import date
from dateutil.relativedelta import relativedelta
sys.path.insert(0, "..")
import CT, FN

# ── Configuration ────────────────────────────────────────────────────────────
API_KEY = 'WGIBI4lbwMEiGOnsCZY7gIVcWweUlVueMjvqE1cB'

FACILITIES = {
	'Alcoa_Allowance_Management_Inc': 6705,
	'Colstrip':						   6076,
	'Gen_J_M_Gavin':				   8102,
	'Hunter':						   6165,
	'Intermountain':				   6481,
	'James_H_Miller_Jr':			   6002,
	'Labadie':						   2103,
	'Martin_Lake':					   6146,
	'Miami_Fort_Power_Station':		   2832,
	'Milton_R_Young':				   2823,
	'New_Madrid_Power_Plant':		   2167,
	'Thomas_Hill_Energy_Center':	   2168,
}

START	   = date(2019, 1, 1)
END		   = date(2019, 3, 31)
OUTPUT_DIR = Path(CT.d_cems)
URL		   = 'https://api.epa.gov/easey/streaming-services/emissions/apportioned/hourly'

DELAY		= 1.0	# seconds between every request
RETRY_WAIT	= 30	# seconds to wait after a 429
MAX_RETRIES = 5		# max retries per request
# ─────────────────────────────────────────────────────────────────────────────

def log_failure(facility_id, name, year, month, reason):
	timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
	line = f"{timestamp} | {year}-{month:02d} | id={facility_id} | {name} | {reason}\n"
	with open(LOG_FILE, 'a') as f:
		f.write(line)

def download_month(facility_id, year, month):
	first_day = date(year, month, 1)
	last_day  = (first_day + relativedelta(months=1)) - relativedelta(days=1)

	params = {
		'api_key':	  API_KEY,
		'facilityId': facility_id,
		'beginDate':  first_day.isoformat(),
		'endDate':	  last_day.isoformat(),
	}

	for attempt in range(1, MAX_RETRIES + 1):
		resp = requests.get(URL, params=params, timeout=120)

		if resp.status_code == 200:
			data = resp.json()
			return pd.DataFrame(data) if data else None

		elif resp.status_code == 429:
			print(f"429 rate limit (attempt {attempt}/{MAX_RETRIES}), waiting {RETRY_WAIT}s ...", flush=True)
			time.sleep(RETRY_WAIT)

		else:
			print(f"HTTP {resp.status_code} — skipping")
			return None

	print(f"Failed after {MAX_RETRIES} retries — skipping")
	return None


def main():
	OUTPUT_DIR.mkdir(exist_ok=True)

	cursor	  = START.replace(day=1)
	end_month = END.replace(day=1)

	while cursor <= end_month:
		year, month = cursor.year, cursor.month

		for name, fid in FACILITIES.items():
			dir_name  = f"{fid}_{name}"						   # e.g. 6705_Alcoa_Allowance_Management_Inc
			file_name = f"{fid}_{name}_{year}_{month:02d}.csv" # e.g. 6705_Alcoa_Allowance_Management_Inc_2019_01.csv

			out_path = OUTPUT_DIR / dir_name / file_name
			out_path.parent.mkdir(parents=True, exist_ok=True)

			if out_path.exists():
				print(f"[{year}-{month:02d}] {dir_name} — already exists, skipping")
				continue

			print(f"[{year}-{month:02d}] {dir_name} ...", end=' ', flush=True)
			df = download_month(fid, year, month)

			if df is not None and not df.empty:
				df.to_csv(out_path, index=False)
				print(f"saved {len(df):,} rows")
			elif df is not None:
				print("no data returned")
				log_failure(fid, name, year, month, f"HTTP {status} — no data returned")
			else:
				reason = f"HTTP {status} — max retries exceeded" if status == 429 else f"HTTP {status}"
				print(f"failed ({reason})")
				log_failure(fid, name, year, month, reason)


			time.sleep(DELAY)

		cursor += relativedelta(months=1)

	print("\nDone.")


main()

import os, sys, glob, IPython, xarray as xr, numpy as np, pandas as pd, math, pyarrow, fnmatch
import pyarrow as pa
import pyarrow.csv as pa_csv
from tqdm import tqdm
from timezonefinder import TimezoneFinder
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, "..")
import CT, FN
from post_plot_vsCEMS import _plot_csf_scatter

## Parameters
CFG={'POST_PRCS_VER':	CT.PRCS_VER['post'],
	 'CSF_PRCS_VER':	CT.PRCS_VER['csf'],
	 'TARGET_INFO':		CT.df_target, 
	 'o_trop':			{'o_read':True, 'type':'csv', 'QF':[('nobs',			25.,	9999.,	'abs'),
															('u_std',			-9999.,	0.8,	'abs'),
															('v_std',			-9999.,	0.8,	'abs'),
															('rsq_detrend',		0.8,	9999.,	'abs'),
															('t_pss',			-9999., 2.0,	'abs'),		# Quality filtering for NOx
															('nox_lifetime',	3.0,	8.0,	'abs')]},
	 'o_cems':			{'o_read':True},
	'CEMS_LOOKBACK_HRS': 6,
	 "transport_bins":	[(0,30), (30,60), (60,90), (90,120), (120,150), (150,180), (180,210), (210,240)],  # minutes
	 "flux_col":		"flux_no2",   # or "flux_no2_O" if pre-suffix CSVs
	 'snapshot':		True,
	}


target=					pd.read_csv(CFG["TARGET_INFO"])
target["facilityId"]=	target["facilityId"].astype(str)
#target=					target.loc[target['facilityId'].isin(['6705', '6076', '8102', '6165', '6481', '6002', '2103', '6146', '2832', '2823', '2167', '2168'])].reset_index(drop=True)
target=					target.loc[target['facilityId'].isin(['2103', '6076'])].reset_index(drop=True)


## Functions
def _prcs_QF(sate, data, QF):
	data['QF_abs']=						0		# 0: good (default), 1: bad
	data['QF_pct']=						0
	for qf in QF:
		col, vmin, vmax, mode = qf[0], qf[1], qf[2], qf[3]
		print (qf)
		if mode == 'abs':
			lBAD=								(data[col] < vmin) | (data[col] > vmax)
			data.loc[lBAD,'QF_abs']=			1
		elif mode == 'pct':
			scope=								qf[4]
			cities=								data['TargetName'].unique() if scope == 'eachcity' else [slice(None)]

			for city in cities:
				subset=							data.loc[city, col] if isinstance(city, slice) else data.loc[data['TargetName'] == city, col]
				_min=							np.sort(subset.values)[int(len(subset) * vmin)]
				_max=							np.sort(subset.values)[int(len(subset) * vmax)-1]
				bad=							(subset < _min) | (subset > _max)
				data.loc[bad.index[bad], 'QF_pct'] = 1
	return data



def _read_gauss(sate, target, CFG):
	"""Read all csf_*.csv files for each facility using PyArrow. Returns long-form DataFrame."""

	base_din  = CT.d_noxno2 + "d_dat/csf/" + CFG["CSF_PRCS_VER"] + "/"
	drop_cols = {"lon_a3_i", "lon_a3_f", "lat_a3_i", "lat_a3_f"}
	pa_read   = pa_csv.ReadOptions(use_threads=True)
	pa_parse  = pa_csv.ParseOptions(delimiter=",")
	pa_conv   = pa_csv.ConvertOptions(strings_can_be_null=False)

	# --- Collect all files ---
	tasks = []
	for _, row in target.iterrows():
		tid, tname = row["facilityId"], row["facilityName"]
		tdir = base_din + f"{tid}_{tname}/".replace(" ", "_")
		try:
			tasks += [(tid, tname, e.path, e.name)
					  for e in os.scandir(tdir)
					  if e.is_file()
					  and fnmatch.fnmatch(e.name, "csf_*.csv")
					  and "_cems" not in e.name]
		except FileNotFoundError:
			print(f"  [WARN] Not found: {tdir}")

	print(f"  [READ] {len(tasks)} files across {target['facilityId'].nunique()} facilities")

	# --- Read one file ---
	def _read_one(tid, tname, path, fname):
		try:
			tbl = pa_csv.read_csv(path, pa_read, pa_parse, pa_conv)
			tbl = tbl.drop([c for c in drop_cols if c in tbl.schema.names])
			for j, f in enumerate(tbl.schema):		# normalise tz-aware timestamps
				if pa.types.is_timestamp(f.type) and f.type.tz:
					tbl = tbl.set_column(j, f.name,
										 tbl.column(f.name).cast(pa.timestamp("us", tz="UTC")))
			n = len(tbl)
			for col, val in [("source_file", fname), ("facilityId", tid), ("facilityName", tname)]:
				tbl = tbl.append_column(col, pa.array([val] * n))
			return tbl
		except Exception as e:
			print(f"  [WARN] {path}: {e}")
			return None

	# --- Parallel read ---
	with ThreadPoolExecutor(max_workers=16) as ex:
		futures = {ex.submit(_read_one, *t): t for t in tasks}
		tables	= [f.result() for f in tqdm(as_completed(futures), total=len(futures),
				   desc="Reading CSF") if f.result() is not None]

	if not tables:
		return pd.DataFrame()

	out = pa.concat_tables(tables, promote_options="default").to_pandas()
	age_col = "age_hours_H" if "age_hours_H" in out.columns else "age_hours"
	out["transport_min"] = pd.to_numeric(out[age_col], errors="coerce") * 60.

	pq_path = base_din + f"{sate}_l2_{CFG['CSF_PRCS_VER']}.parquet"
	out.to_parquet(pq_path, engine="pyarrow", index=False)
	print(f"  [READ] {len(out)} rows saved → {pq_path}")
	return out




def _read_cems(target, trop_l2, CFG):
	"""
	Read raw CEMS CSV files and sample rows within lookback window for each
	TROPOMI overpass in trop_l2. Reuses _read_cems_nox logic per facility.
	Returns target (with facility-level stats) and cems_mean (per overpass).
	"""
	tf = TimezoneFinder()

	# One row per unique overpass, with parsed overpass time
	overpasses=						(trop_l2[["facilityId", "source_file"]].drop_duplicates().reset_index(drop=True))
	overpasses["t_overpass"]=		pd.to_datetime(overpasses["source_file"].str.split("_").str[9])#, format="%Y%m%d%H%M", errors="coerce")
	overpasses=						overpasses.dropna(subset=["t_overpass"]).reset_index(drop=True)

	results = []

	for _, trow in target.iterrows():
		tid=	trow["facilityId"]
		tname=	trow["facilityName"]
		tname_long = f"{tid}_{tname}".replace(" ", "_")

		# --- Read all raw CEMS files for this facility ---
		# Only read months spanned by this facility's overpasses (± lookback)
		ops_fac  = overpasses.loc[overpasses["facilityId"] == tid, "t_overpass"]
		if ops_fac.empty:
			continue
		t_min	 = ops_fac.min() - pd.Timedelta(hours=CFG["CEMS_LOOKBACK_HRS"])
		t_max	 = ops_fac.max()
		months	 = pd.date_range(t_min.to_period("M").to_timestamp(),
								 t_max.to_period("M").to_timestamp(), freq="MS")
		files	 = [os.path.join(CT.d_cems, tname_long, f"{tname_long}_{m.year}_{m.month:02d}.csv")
					for m in months]
		files	 = [f for f in files if os.path.exists(f)]
		if not files:
			print(f"  [CEMS] No files found for {tname_long}")
			continue
		cems = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

		# --- Parse local → UTC (same as _read_cems_nox) ---
		cems["tstmp_local"] = pd.to_datetime(cems["date"]) + pd.to_timedelta(cems["hour"], unit="h")
		tz = tf.timezone_at(lat=trow["lat"], lng=trow["lon"]) or "UTC"
		cems["tstmp_utc"] = (cems["tstmp_local"]
							 .dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
							 .dt.tz_convert("UTC")
							 .dt.tz_localize(None))
		cems["noxMass"] = pd.to_numeric(cems["noxMass"], errors="coerce")

		# Sum all units per hour
		cems_hourly = (cems.groupby("tstmp_utc", as_index=False)["noxMass"]
						   .sum()
						   .rename(columns={"noxMass": "noxMass_tph"}))
		cems_hourly["noxMass_tph_metric"] = cems_hourly["noxMass_tph"] * 0.000453592

		# --- Sample lookback window per overpass ---
		for _, op in overpasses.loc[overpasses["facilityId"] == tid].iterrows():
			t_end	= op["t_overpass"]
			t_start = t_end - pd.Timedelta(hours=CFG["CEMS_LOOKBACK_HRS"])
			window	= cems_hourly.loc[(cems_hourly["tstmp_utc"] >= t_start) &
									  (cems_hourly["tstmp_utc"] <= t_end)].copy()
			if window.empty:
				continue
			window["facilityId"]   = tid
			window["facilityName"] = tname
			window["source_file"]  = op["source_file"]
			results.append(window)

	if not results:
		print("  [CEMS] No matching windows found.")
		return target, pd.DataFrame()

	cems_long = pd.concat(results, ignore_index=True)

	# Per-overpass mean → for scatter plot merge
	cems_mean = (cems_long
				 .groupby(["facilityId", "source_file"])["noxMass_tph_metric"]
				 .mean()
				 .reset_index()
				 .rename(columns={"noxMass_tph_metric": "cems_nox_mean"}))

	# Facility-level stats → attach to target
	cems_fac = (cems_long
				.groupby("facilityId")["noxMass_tph_metric"]
				.agg(cems_nox_fac_mean="mean", cems_nox_fac_std="std")
				.reset_index())
	target = target.merge(cems_fac, on="facilityId", how="left")

	# Save
	base_din = CT.d_noxno2 + "d_dat/csf/" + CFG["CSF_PRCS_VER"] + "/"
	cems_long.to_parquet(base_din + f"cems_long_{CFG['CSF_PRCS_VER']}.parquet", engine="pyarrow", index=False)
	cems_mean.to_parquet(base_din + f"cems_mean_{CFG['CSF_PRCS_VER']}.parquet", engine="pyarrow", index=False)
	print(f"  [CEMS] {len(cems_long)} sampled rows, {cems_long['facilityId'].nunique()} facilities")

	return target, cems_mean



## Run
if CFG["o_trop"]["o_read"]:
	if CFG["o_trop"]["type"] == "csv":
		trop_l2 = _read_gauss(sate="trop", target=target, CFG=CFG)
	if CFG["o_trop"]["type"] == "pq":
		trop_l2 = pd.read_parquet(f"{CT.d_noxno2}d_dat/csf/{CFG['CSF_PRCS_VER']}/trop_l2_{CFG['POST_PRCS_VER']}.parquet", engine="pyarrow" )

trop_l2=			_prcs_QF(sate='trop', data=trop_l2, QF=CFG['o_trop']['QF'])
trop_l2=			trop_l2.loc[trop_l2['QF_abs']==0]

if CFG["o_cems"]["o_read"]:
	target, cems_mean = _read_cems(target=target, trop_l2=trop_l2, CFG=CFG)


_plot_csf_scatter(trop_l2, cems_mean, target, CFG)






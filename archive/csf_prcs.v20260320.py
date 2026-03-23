# =============================================================================
# csf_prcs.py  —  Cross-Sectional Flux (CSF) processing
# =============================================================================
# --- Standard library ---
import os, sys, glob, math
from datetime import timedelta, date
from concurrent.futures import ThreadPoolExecutor, as_completed
# --- Third-party: data / science ---
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from scipy.optimize import curve_fit
import sklearn.metrics
# --- Third-party: geo ---
import pyproj
import geopy.distance
from shapely.geometry import Point, Polygon, MultiPolygon, shape, box
from timezonefinder import TimezoneFinder
from netCDF4 import Dataset
# --- Third-party: viz ---
import matplotlib as mpl
import IPython
# --- Local ---
sys.path.insert(0, "..")
import CT, FN
from csf_plot import _plot_csf

# =============================================================================
# CONFIG  —  edit here to change run behaviour
# =============================================================================

CFG = {
	"CSF_PRCS_VER":		CT.PRCS_VER["csf"],
	"TARGET_INFO":		CT.df_target,
	"SATE_INFO":		["trop", CT.DATA_VER["trop"], CT.PRCS_VER["trop"], "2022-10-05", "2022-10-06", "qf_good", "prcsd"],
	"HYSTRAJ_RUN_VER":	CT.PRCS_VER["hystraj"],
	"o_calc_nox":		False,
	"o_plot":			True,
	"snapshot":			True,
	"CEMS_LOOKBACK_HRS": 6
}

target=					pd.read_csv(CFG["TARGET_INFO"])
target["facilityId"]=	target["facilityId"].astype(str)
target=					target.loc[target["facilityId"].isin(["6705"])].reset_index(drop=True)


# Gaussian curve-fitting iteration ceiling
MAXFEV = 399_999

# Earth radius used in haversine [m]
EARTH_RADIUS_M = 6_371_000.0

# Orthogonal transect half-length [degrees]
TRANSECT_HALFLENGTH = 1.0

# Distance threshold for matching satellite pixels to transect [degrees]
DIST_THRESH_TROP = 0.06 #0.08


# =============================================================================
# GEOMETRY HELPERS
# =============================================================================

def _haversine_m(lat1, lon1, lat2, lon2):
	"""Vectorised haversine distance; returns metres."""
	dlat = np.radians(lat2 - lat1)
	dlon = np.radians(lon2 - lon1)
	a = (np.sin(dlat / 2) ** 2
		 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
	return EARTH_RADIUS_M * 2 * np.arcsin(np.sqrt(a))


def _calculate_bearing(lat1, lon1, lat2, lon2):
	"""Bearing from point-1 → point-2, degrees [0, 360)."""
	lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
	dlon = lon2 - lon1
	y = np.sin(dlon) * np.cos(lat2)
	x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
	return (np.degrees(np.arctan2(y, x)) + 360) % 360


# =============================================================================
# DATA READERS
# =============================================================================
def _read_tdump(satellite, tname_long, tid, time):
	"""
	Read a HYSPLIT tdump file and return a DataFrame with trajectory info,
	wind direction, and orthogonal wind-speed components.
	Returns empty string on file-not-found.
	"""
	d_tdump = CT.d_noxno2 + "d_dat/hysplit/" + CFG["HYSTRAJ_RUN_VER"] + "/" + tname_long + "/tdump/"
	f_tdump = f"tdump_{satellite}_{tid}_*_{time}_{CFG['HYSTRAJ_RUN_VER']}"
	matches = glob.glob(d_tdump + f_tdump)

	if not matches:
		with open("./csf_prcs.log", "a") as log:
			log.write(f"[tdump not found] d_tdump={d_tdump} | f_tdump={f_tdump}\n")
		return ""

	hdrs = ["?1", "?2", "year", "month", "day", "hour", "minute",
			"forecast_hour", "age_hours", "latitude", "longitude",
			"z_magl", "pres", "temp"]

	with open(matches[0]) as fh:
		nmet_ctr = int(fh.readline().strip().split()[0])

	tdump = (pd.read_csv(matches[0], skiprows=4 + nmet_ctr, names=hdrs,
						 sep=r"\s+", engine="python")
			   .reset_index(drop=True))

	tdump["tdump_id"] = np.arange(len(tdump))
	tdump["tstmp"] = [
		pd.to_datetime(
			f"20{tdump.loc[i,'year']}-{tdump.loc[i,'month']:02d}-"
			f"{tdump.loc[i,'day']:02d} {tdump.loc[i,'hour']:02d}:"
			f"{tdump.loc[i,'minute']:02d}"
		)
		for i in range(len(tdump))
	]

	lats, lons = tdump["latitude"].values, tdump["longitude"].values

	# Wind direction: bearing from start to each downwind coordinate
	bearings = _calculate_bearing(lats[0], lons[0], lats, lons)
	bearings[0] = bearings[1]
	tdump["wd"] = bearings

	# Wind-speed components
	dt = (tdump["tstmp"][2] - tdump["tstmp"][1]).seconds
	ws_ortho = _haversine_m(lats[:-1], lons[:-1], lats[1:],  lons[1:]) / dt
	u		 = _haversine_m(lats[:-1], lons[:-1], lats[:-1], lons[1:]) / dt
	v		 = _haversine_m(lats[:-1], lons[1:],  lats[1:],  lons[1:]) / dt

	tdump["wso"] = np.append(ws_ortho, -9999.0)
	tdump["u"]	 = np.append(u,		   -9999.0)
	tdump["v"]	 = np.append(v,		   -9999.0)

	return tdump


def _read_trop(dfin, tname_long, **isel_kwargs):
	"""
	Read a TROPOMI NetCDF file; return a (Geo)DataFrame of valid pixels.
	Pass scanline/pixel slice kwargs via isel_kwargs if needed.
	"""
	var_list = [
		"nitrogendioxide_tropospheric_column",
		"nitrogendioxide_tropospheric_column_precision",
		"qa_value", "time_utc", "latitude", "longitude",
	]
	if CFG["o_plot"]:
		var_list += ["longitude_bounds", "latitude_bounds"]

	with xr.open_dataset(dfin, engine="netcdf4", mask_and_scale=False) as ds:
		data = ds[var_list]
		if isel_kwargs:
			data = data.isel(**isel_kwargs)
		data = data.load()

	no2		 = data["nitrogendioxide_tropospheric_column"].values
	no2_prec = data["nitrogendioxide_tropospheric_column_precision"].values
	qa		 = data["qa_value"].values
	time_utc = data["time_utc"].values
	lat		 = data["latitude"].values
	lon		 = data["longitude"].values

	# Broadcast time_utc to pixel dimension
	time_utc = np.repeat(time_utc[:, :, np.newaxis], no2.shape[2], axis=2)

	mask = no2.ravel() >= 0.0
	if CFG["SATE_INFO"][5] == "qf_good":
		mask &= qa.ravel() >= 0.5

	trop = pd.DataFrame({
		"no2":			 no2.ravel()[mask] * 1e6,	# mol m-2 --> umol m-2
		"no2_precision": no2_prec.ravel()[mask],
		"qa_value":		 qa.ravel()[mask],
		"time_utc":		 time_utc.ravel()[mask],
		"latitude":		 lat.ravel()[mask],
		"longitude":	 lon.ravel()[mask],
	})

	if CFG["o_plot"]:
		lonb = data["longitude_bounds"].values.reshape(-1, 4)[mask]
		latb = data["latitude_bounds"].values.reshape(-1, 4)[mask]

		trop = gpd.GeoDataFrame(
			trop, crs="EPSG:4326",
			geometry=gpd.points_from_xy(trop.longitude, trop.latitude),
		)

		finite = np.isfinite(lonb).all(axis=1) & np.isfinite(latb).all(axis=1)
		coords	= np.stack([lonb, latb], axis=2)			# (N, 4, 2)
		closed	= np.concatenate([coords, coords[:, :1, :]], axis=1)  # (N, 5, 2)
		polys	= np.where(finite, [Polygon(c) for c in closed], None)

		trop = (trop.assign(geom_poly=polys)
					.set_geometry("geom_poly", crs="EPSG:4326"))

	return trop


def _build_true_tdump(data_csf, tdump):
	"""
	Step 4: Build 'true' trajectory by connecting a3 lat/lon coordinates
	and compute true wind speed components from adjacent a3 distances.
	Returns a tdump-compatible DataFrame, or empty DataFrame if < 2 valid a3 points.
	"""
	valid = data_csf.dropna(subset=["lon_a3", "lat_a3"]).reset_index(drop=True)
	if len(valid) < 2:
		return pd.DataFrame()

	true = pd.DataFrame({
		"tdump_id":  valid["tdump_id"].values,
		"longitude": valid["lon_a3"].astype(float).values,
		"latitude":  valid["lat_a3"].astype(float).values,
		"age_hours": valid["age_hours"].values,
		"month":	 valid["month"].values,
		"hour":		 valid["hour"].values,
		"temp":		 valid["temp"].values,
		"pres":		 valid["pres"].values,
		"tstmp":	 tdump.loc[tdump["tdump_id"].isin(valid["tdump_id"]), "tstmp"].values,
	}).reset_index(drop=True)

	lats, lons = true["latitude"].values, true["longitude"].values
	dt		   = (tdump.loc[1, "tstmp"] - tdump.loc[0, "tstmp"]).seconds

	wso = _haversine_m(lats[:-1], lons[:-1], lats[1:],	lons[1:])  / dt
	u	= _haversine_m(lats[:-1], lons[:-1], lats[:-1], lons[1:])  / dt
	v	= _haversine_m(lats[:-1], lons[1:],  lats[1:],	lons[1:])  / dt

	true["wso"] = np.append(wso, wso[-1])
	true["u"]	= np.append(u,	 u[-1])
	true["v"]	= np.append(v,	 v[-1])

	bearings	   = _calculate_bearing(lats[0], lons[0], lats, lons)
	bearings[0]    = bearings[1]
	true["wd"]	   = bearings

	return true


# --- Drop-in function (add near other readers) ---
def _read_cems_nox(tid, tname, time_utc):
	"""
	Read CEMS hourly NOx mass [short tons] for a facility, summed across all
	units, for the 6-hour window ending at the TROPOMI overpass time (UTC).

	CEMS 'hour' column is the START of the hour in LOCAL time, stored as an
	integer 0–23. We convert to UTC using the facility's timezone, then filter
	to [time_utc - lookback, time_utc].

	Returns a DataFrame with columns: [tstmp_utc, noxMass_tph] where
	noxMass_tph is total facility NOx in short tons hr⁻¹.
	Returns empty DataFrame on missing file or no data in window.
	"""

	# --- Find CEMS files for this facility (may span multiple months) ---
	pattern = os.path.join(CT.d_cems+tname, f"{tname}_*.csv")
	files	= sorted(glob.glob(pattern))
	if not files:
		print(f"  [CEMS] No files found: {pattern}")
		return pd.DataFrame()

	cems = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

	# --- Parse local timestamp ---
	cems["tstmp_local"] = pd.to_datetime(cems["date"]) + pd.to_timedelta(cems["hour"], unit="h")

	# --- Get facility timezone from its coordinates ---
	tf	= TimezoneFinder()
	row = pd.read_csv(CFG["TARGET_INFO"])
	row = row.loc[row["facilityId"].astype(str) == str(tid)].iloc[0]
	tz	= tf.timezone_at(lat=row["lat"], lng=row["lon"])
	if tz is None:
		print(f"  [CEMS] Could not determine timezone for {tid}, assuming UTC")
		tz = "UTC"

	# --- Convert local → UTC ---
	cems["tstmp_utc"] = (
		cems["tstmp_local"]
		.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
		.dt.tz_convert("UTC")
		.dt.tz_localize(None)			# strip tzinfo for comparison
	)

	# --- Filter to lookback window ---
	t_end	= pd.Timestamp(time_utc).tz_convert("UTC").tz_localize(None)
	t_start = t_end - pd.Timedelta(hours= CFG["CEMS_LOOKBACK_HRS"])
	cems	= cems[(cems["tstmp_utc"] >= t_start) & (cems["tstmp_utc"] <= t_end)].copy()

	if cems.empty:
		print(f"  [CEMS] No data in window {t_start} – {t_end} for {tid}")
		return pd.DataFrame()

	# --- Sum all units per hour, convert lbs → metric tons (×0.00045) ---
	cems["noxMass"] = pd.to_numeric(cems["noxMass"], errors="coerce")
	hourly = (cems.groupby("tstmp_utc", as_index=False)["noxMass"].sum().rename(columns={"noxMass": "noxMass_tph"}))
	hourly["noxMass_tph_metric"] = hourly["noxMass_tph"] * 0.000453592	 # lbs/hr → metric tons_NOx hr-1



	return hourly

# =============================================================================
# SAMPLING & CONSTRAINTS
# =============================================================================

def _define_gaussian_constraints(satellite, data_sampled, trop_csf):
	"""Return per-transect Gaussian fitting bounds."""
	if satellite == "trop":
		n = data_sampled["tdump_id"].nunique()
		c = pd.DataFrame({"tdump_id": data_sampled["tdump_id"].unique()})
		defaults = {
			"a0_min": 0.,	 "a0_max": 500.,
			"a1_min": -50.,  "a1_max": 50.,
			"a2_min": 0.,	 "a2_max": 1000.,
			"a3_min": -100., "a3_max": 100.,
			"a4_min": 1.,	 "a4_max": 100.,
			"dtdump_id_min": 0.,
		}
		for col, val in defaults.items():
			c[col] = val
		return c


def _sample_soundings_along_traj(satellite, data, tdump):
	"""
	For each trajectory segment, build an orthogonal transect and
	collect satellite pixels within DIST_THRESH of that transect.
	Returns a DataFrame (empty if nothing sampled).
	"""
	if satellite == "trop":
		dist_thresh = DIST_THRESH_TROP
		val_col		= "no2"

	dsec	   = (tdump.loc[1, "tstmp"] - tdump.loc[0, "tstmp"]).seconds
	data_lon   = data["longitude"].values
	data_lat   = data["latitude"].values
	data_val   = data[val_col].values
	lons, lats = tdump["longitude"].values, tdump["latitude"].values

	results = []
	for i in range(len(tdump) - 1):
		# Midpoint of segment
		lon = (lons[i] + lons[i + 1]) / 2
		lat = (lats[i] + lats[i + 1]) / 2
		dlon = lons[i + 1] - lons[i]
		dlat = lats[i + 1] - lats[i]
		slope = -1.0 / ((dlat / (dlon + 1e-99)) + 1e-99)

		ws = _haversine_m(lats[i], lons[i],
						  np.array([lats[i + 1]]),
						  np.array([lons[i + 1]]))[0] / dsec

		# Orthogonal transect coordinates
		if abs(slope) > 1:
			lat_tmp = np.arange(lat - TRANSECT_HALFLENGTH, lat + TRANSECT_HALFLENGTH, 0.01)
			lon_tmp = (lat_tmp - lat) / slope + lon
		else:
			lon_tmp = np.arange(lon - TRANSECT_HALFLENGTH, lon + TRANSECT_HALFLENGTH, 0.01)
			lat_tmp = slope * (lon_tmp - lon) + lat

		mid		 = len(lon_tmp) // 2
		dist_tmp = _haversine_m(lat_tmp[mid], lon_tmp[mid], lat_tmp, lon_tmp) / 1e3   # m → km
		ref		 = lon_tmp if lon_tmp.max() != lon_tmp.min() else lat_tmp
		dist_tmp = np.where(ref < ref[mid], -dist_tmp, dist_tmp)

		# Sample pixels near transect
		ddlon  = data_lon[np.newaxis, :] - lon_tmp[:, np.newaxis]
		ddlat  = data_lat[np.newaxis, :] - lat_tmp[:, np.newaxis]
		nearby = (ddlon ** 2 + ddlat ** 2) < dist_thresh ** 2
		hit_rows = np.where(nearby.any(axis=1))[0]

		if len(hit_rows) < 3:
			continue

		rows = []
		for j in hit_rows:
			idx = np.where(nearby[j])[0]
			rows.append({
				"lon_ortho":   lon_tmp[j],
				"lat_ortho":   lat_tmp[j],
				"dist_ortho":  dist_tmp[j],
				val_col:	   data_val[idx].mean(),
				"tdump_id":    i,
				"lon_sampled": data_lon[idx].mean(),
				"lat_sampled": data_lat[idx].mean(),
			})

		if len(rows) >= 3:
			results.append(pd.DataFrame(rows))

	if not results:
		return pd.DataFrame()

	sampled = pd.concat(results, ignore_index=True)
	return sampled[sampled.groupby("tdump_id")["tdump_id"].transform("size") >= 3]


# =============================================================================
# CSF CALCULATION
# =============================================================================
def _calc_csf(satellite, target, data, constraint, tdump):
	"""
	Fit a Gaussian to each orthogonal transect's satellite profile and
	compute cross-sectional flux (CSF) metrics.
	"""
	merge_cols = ["tdump_id", "longitude", "latitude", "age_hours", "month", "hour", "temp", "pres", "u", "v", "wso", "wd"]
	out = (pd.DataFrame({"tdump_id": data["tdump_id"].unique()})
			 .merge(tdump[merge_cols], on="tdump_id", how="left")
			 .rename(columns={"longitude": "lon_tdump", "latitude": "lat_tdump"}))
	out["month"] = out["month"].values.astype(int)

	voi = "no2" if satellite == "trop" else None

	for i in range(len(out)):
		bounds = (
			[constraint.loc[i, f"a{k}_min"] for k in range(5)],
			[constraint.loc[i, f"a{k}_max"] for k in range(5)],
		)
		out.loc[i, "dtdump_id_min"] = constraint.loc[i, "dtdump_id_min"]

		tmp = data.loc[data["tdump_id"] == out.loc[i, "tdump_id"]].reset_index(drop=True)
		if len(tmp) <= 1:
			continue

		x, y = tmp["dist_ortho"].values, tmp[voi].values
		popt, pcov = curve_fit(FN._gaussian, x, y, bounds=bounds, maxfev=MAXFEV)
		a0, a1, a2, a3, a4 = popt
		perr = np.sqrt(np.diag(pcov))

		# Peak / spread geometry
		tmp["del_dist_ortho"] = np.abs(tmp["dist_ortho"].values - a3)
		out.loc[i, "lon_a3"] = tmp.loc[tmp["del_dist_ortho"].idxmin(), "lon_ortho"]
		out.loc[i, "lat_a3"] = tmp.loc[tmp["del_dist_ortho"].idxmin(), "lat_ortho"]

		if CFG["o_plot"]:
			for suffix, offset in [("i", -(a4 / 2)), ("f", +(a4 / 2))]:
				tmp[f"del_dist_{suffix}"] = np.abs(tmp["dist_ortho"].values - (a3 + offset))
				out.loc[i, f"lon_a3_{suffix}"] = tmp.loc[tmp[f"del_dist_{suffix}"].idxmin(), "lon_ortho"]
				out.loc[i, f"lat_a3_{suffix}"] = tmp.loc[tmp[f"del_dist_{suffix}"].idxmin(), "lat_ortho"]

		# Store fit params & uncertainties
		for k, name in enumerate(["a0", "a1", "a2", "a3", "a4"]):
			out.loc[i, name]			  = popt[k]
			out.loc[i, name + "sig"]	  = perr[k]
			out.loc[i, name + "sigpct"]   = perr[k] / popt[k] * 100.0

		# Quality metrics
		sigma	   = a4 / np.sqrt(8.0 * np.log(2))
		width_2sig = sigma * 4.0
		gaps	   = np.abs(np.diff(tmp["dist_ortho"].values))

		out.loc[i, "d_gap"]    = gaps.max() / width_2sig * 100.0
		out.loc[i, "d_right"]  = (tmp["dist_ortho"].max() - (a3 + sigma * 2)) / width_2sig * 100.0
		out.loc[i, "d_left"]   = ((a3 - sigma * 2) - tmp["dist_ortho"].min()) / width_2sig * 100.0
		out.loc[i, "d_center"] = np.abs(tmp["dist_ortho"] - a3).min() / width_2sig * 100.0
		out.loc[i, "nobs"]		 = len(tmp)
		out.loc[i, "nobs_left"]  = (tmp["dist_ortho"] < a3).sum()
		out.loc[i, "nobs_right"] = (tmp["dist_ortho"] > a3).sum()

		y_fit = FN._gaussian(tmp["dist_ortho"], *popt)
		out.loc[i, "mae"]  = np.abs(tmp[voi].values - y_fit).mean()
		out.loc[i, "mape"] = sklearn.metrics.mean_absolute_percentage_error(tmp[voi].values, y_fit)
		out.loc[i, "rsq"]  = sklearn.metrics.r2_score(tmp[voi].values, y_fit)

		trend	   = a0 + a1 * tmp["dist_ortho"].values
		out.loc[i, "rsq_detrend"] = sklearn.metrics.r2_score(
			tmp[voi].values - trend, y_fit - trend
		)

		if satellite == "trop":
			# CSF: µmol NO₂/m² → tNO₂/hr
			out.loc[i, "flux_no2"] = (
				0.5 * (np.pi / np.log(2)) ** 0.5
				* (a2 / 1e6 * 46.0055 / 1e6)	# µmol_NO2 m-2 → t_NO2 m-2
				* (a4 * 1e3)					# km → m
				* out.loc[i, "wso"]				# m s-1
				* 3600.0
			)

	# Wind statistics and direction comparison
	if "flux_no2" in out.columns or "flux_co2" in out.columns:
		out["u_std"] = out["u"].std()
		out["v_std"] = out["v"].std()
		out["wd_diff"] = None
		if "lat_a3" in out.columns and len(out) > 1:
			wd_a3	= _calculate_bearing(out["lat_a3"].iloc[0],  out["lon_a3"].iloc[0], out["lat_a3"].iloc[1:].to_numpy(),	out["lon_a3"].iloc[1:].to_numpy())
			wd_traj = _calculate_bearing(out["lat_tdump"].iloc[0], out["lon_tdump"].iloc[0], out["lat_tdump"].iloc[1:].to_numpy(), out["lon_tdump"].iloc[1:].to_numpy())
			wd_diff = (np.asarray(wd_a3) - np.asarray(wd_traj) + 180.0) % 360.0 - 180.0
			out["wd_diff"] = np.insert(wd_diff, 0, wd_diff[0])


	return out


# =============================================================================
# MAIN PROCESSING LOOP
# =============================================================================

def _build_output_dirs(dout_csv, dout_png):
	os.makedirs(dout_csv, exist_ok=True)
	os.makedirs(dout_png, exist_ok=True)


def _collect_satellite_files(tname):
	"""Return a DataFrame of TROPOMI files and overpass times for the target."""
	d_sate = CT.d_trop_target + CFG["SATE_INFO"][2] + "/d_nc/" + tname + "/"
	f_sate = sorted(glob.glob(d_sate + f"*_{tname}_{CFG['SATE_INFO'][2]}.nc"))

	rows = []
	for f in f_sate:
		with Dataset(f, "r") as ds:
			rows.append({
				"time_utc": ds.variables["time_utc"][0, 0],
				"file":		f.split("/")[-1],
				"dfin":		f,
			})

	sate = pd.DataFrame(rows)
	sate["time_utc"] = pd.to_datetime(sate["time_utc"])
	IPython.embed()
	return sate[
		(sate["time_utc"] >= CFG["SATE_INFO"][3]) &
		(sate["time_utc"] <= CFG["SATE_INFO"][4])
	].reset_index(drop=True)


def _process_overpass(it, i_sate, target, sate, tname, tid, dout_csv, dout_png):
	"""Process one satellite overpass for one target facility."""
	row  = sate.loc[i_sate]
	tstr = row["time_utc"].strftime("%Y%m%d%H%M")
	fout = f"csf_{row['file']}_{CFG['CSF_PRCS_VER']}"

	print(f"  [{it}/{len(target)-1}] {target.loc[it,'facilityId']}	"
		  f"overpass {i_sate}/{len(sate)-1}")

	trop	  = _read_trop(row["dfin"], tname)
	tdump	  = _read_tdump("trop", tname, tid, tstr)

	if len(tdump) <= 1:
		return

	trop_sampled = _sample_soundings_along_traj("trop", trop, tdump)

	if len(trop_sampled) == 0:
		return

	# --- First-pass CSF (steps 1-3), suffix _H for HYSPLIT ---
	constraint	= _define_gaussian_constraints("trop", trop_sampled, "")
	trop_csf	= _calc_csf("trop", target.loc[it], trop_sampled, constraint, tdump)
	trop_csf	= trop_csf.add_suffix("_H").rename(columns={"tdump_id_H": "tdump_id"})

	# --- Optimized CSF (steps 4-7), suffix _O for Optimized ---
	true_tdump		= _build_true_tdump(trop_csf.rename(columns=lambda c: c.replace("_H", "")), tdump)

	if len(true_tdump) >= 2:
		trop_true_sampled = _sample_soundings_along_traj("trop", trop, true_tdump)

		if len(trop_true_sampled) > 0:
			constraint_true = _define_gaussian_constraints("trop", trop_true_sampled, "")
			csf_true		= _calc_csf("trop", target.loc[it], trop_true_sampled, constraint_true, true_tdump)
			csf_true		= csf_true.add_suffix("_O").rename(columns={"tdump_id_O": "tdump_id"})

			# Merge into single DataFrame on tdump_id
			trop_csf = trop_csf.merge(csf_true, on="tdump_id", how="left")

	trop_csf.to_csv(dout_csv + fout + ".csv", index=False)

	# --- CEMS ---
	cems_nox = _read_cems_nox(tid, tname, row["time_utc"])
	if not cems_nox.empty:
		cems_nox.to_csv(dout_csv + fout + "_cems.csv", index=False)
		print(f"  [CEMS] {len(cems_nox)} hourly records, "
			  f"mean NOx = {cems_nox['noxMass_tph_metric'].mean():.2f} tNOx/hr (metric)")

	# --- Plot ---
	if CFG["o_plot"]:
		_plot_csf(
			CFG, target=target.loc[it],
			data=trop, data_sampled=trop_sampled, data_true_sampled=trop_true_sampled,
			data_csf=trop_csf, tdump=tdump,
			sate=row, flux_unit="[tNO2/hr]",
			o_plot_wddiff=False, dfout=dout_png + fout,
			cems_nox=cems_nox,
			true_tdump=true_tdump,
		)


def _csf_prcs(CFG, target):
	"""Main entry point: loop targets → overpasses → compute CSF."""
	for it in range(len(target)):
		tid   = target.loc[it, "facilityId"]
		tname = (tid + "_" + target.loc[it, "facilityName"]).replace(" ", "_")

		dout_csv = CT.d_noxno2 + "d_dat/csf/" + CFG["CSF_PRCS_VER"] + "/" + tname + "/"
		dout_png = CT.d_noxno2 + "d_fig/csf/" + CFG["CSF_PRCS_VER"] + "/" + tname + "/"
		_build_output_dirs(dout_csv, dout_png)

		print(f"Processing target {it}/{len(target)-1}: {tid} {target.loc[it,'facilityName']}")

		sate = _collect_satellite_files(tname)

		for i_sate in range(0, len(sate)):
			_process_overpass(it, i_sate, target, sate, tname, tid, dout_csv, dout_png)


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
	_csf_prcs(CFG, target)

	if CFG["snapshot"]:
		FN._SNAPSHOT(
			name="csf_prcs",
			tag=CFG["CSF_PRCS_VER"],
			scripts=["csf_prcs.py"],
			cfg=CFG,
			out_dir=os.path.join(CT.d_noxno2, "d_history"),
		)

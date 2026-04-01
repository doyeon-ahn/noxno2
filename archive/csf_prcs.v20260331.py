# =============================================================================
# csf_prcs.py	Cross-Sectional Flux (CSF) processing	v20260324
# =============================================================================
import os, sys, glob
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from scipy.optimize import curve_fit
import sklearn.metrics
from shapely.geometry import Polygon
from timezonefinder import TimezoneFinder
import IPython
sys.path.insert(0, os.pardir)
import CT, FN
from csf_plot import _plot_csf
from csf_nox import run_nox_workflow

# =============================================================================
# CONFIG  —  all tunable parameters live here
# =============================================================================
CFG = {
	# ------------------------------------------------------------------
	# Versioning & paths
	# ------------------------------------------------------------------
	"CSF_PRCS_VER":		CT.PRCS_VER["csf"],
	"TARGET_INFO":		CT.df_target,
	"SATE_INFO":		["trop", CT.DATA_VER["trop"], CT.PRCS_VER["trop"],
						 "2023-01-02", "2024-10-31",	# date range [start, end]
						 #"2019-01-01", "2025-12-31",	# date range [start, end]
						 #"2023-01-30", "2023-01-31",	# date range [start, end]
						 "qf_good", "prcsd"],
	"HYSTRAJ_RUN_VER":	CT.PRCS_VER["hystraj"],

	# ------------------------------------------------------------------
	# Target facility filter  (set to None to process all targets)
	# ------------------------------------------------------------------
	"TARGET_IDS_ALL":	["6705","6076","8102","6165","6481","6002",
						 "2103","6146","2832","2823","2167","2168"],
	"TARGET_IDS_RUN":	['2167'], #'6165'], #"6076"],#, "2103"],			# subset to actually process

	# ------------------------------------------------------------------
	# Satellite pixel QA
	# ------------------------------------------------------------------
	"QA_MIN":			0.5,				# minimum qa_value to keep pixel

	# ------------------------------------------------------------------
	# Trajectory sampling
	# ------------------------------------------------------------------
	"TRANSECT_HALFLENGTH":	100.0,			# orthogonal transect half-length [km]
	"SAMPLE_MODE":	"nearest",				# "mean_within_range": avg pixels within DIST_THRESH_TROP | "nearest": closest pixel per transect point
	"DIST_THRESH_TROP":		6.5,			# pixel-to-transect match radius [km] (used for mean_within_range)
	"MIN_PIXELS_PER_TRANSECT": 3,			# discard transect if fewer pixels (used for mean_within_range)

	# ------------------------------------------------------------------
	# Gaussian curve fitting
	# ------------------------------------------------------------------
	"MAXFEV":			399_999,			# curve_fit iteration ceiling
	"GAUSS_BOUNDS": {						# per-parameter [min, max]
		"a0": [0.,	  500.],				# baseline offset
		"a1": [-50.,  50. ],				# baseline slope
		"a2": [0.,	  1000.],				# amplitude
		"a3": [-50., 50.],					# centre offset [km]
		"a4": [1.,	  40.],					# FWHM [km]
	},

	# ------------------------------------------------------------------
	# Gaussian quality flag  (QF_gauss_abs: 0=good, 1=bad)
	# ------------------------------------------------------------------
	"QF": {
		"rsq_detrend_min":	0.0,			# detrended R²
		"d_gap_max":		50.,			# max gap within FWHM [%]
		"d_left_min":	   -20.,			# left coverage [%]
		"d_right_min":	   -20.,			# right coverage [%]
		"d_center_max":		30.,			# centre sampling [%]
		"a2sigpct_max":		50.,			# amplitude uncertainty [%]
		"a4sigpct_max":		50.,			# width uncertainty [%]
		"nobs_min":			5,				# minimum pixel count
	},

	# ------------------------------------------------------------------
	# Plume trajectory optimization using a3
	# ------------------------------------------------------------------
	'TDOPT': {'method':		'moore',
				# High-NO2 pixel-chain trajectory  (tdump_highNO2)
				"search_radius_deg": 0.15,	 # neighbour search radius [deg] (~2–3 TROPOMI pixels)
				#"bg_snr_thresh":	 1.0,	 # at each step move to the unvisited 3x3 Moore neighbour with the highest NO2 still above (zbg + bg_snr_thresh * zbg_sig).
				"bg_snr_thresh":	 0.5,	 # stop threshold: zbg + thresh * zbg_sig
				"max_steps":		 600,	 # hard ceiling on chain length
				"smooth_poly_deg":	2,
			},

	# ------------------------------------------------------------------
	# NOx workflow	(csf_nox.run_nox_workflow)
	# ------------------------------------------------------------------
	"NOX_SUFFIX":		"_O",			   # trajectory suffix to use ("_H" or "_O")
	"NOX_OPTION":		"A",			  # decay option: "A"=empirical, "B"=analytical
	"NOX_RESIDUAL_TOL": 2.0,			  # log-normal fit outlier tolerance (step 2)

	# ------------------------------------------------------------------
	# CEMS
	# ------------------------------------------------------------------
	"CEMS_LOOKBACK_HRS": 6,				  # hours before overpass to include

	# ------------------------------------------------------------------
	# Output & plotting
	# ------------------------------------------------------------------
	"FLUX_UNIT":		"[tNO2/hr]",
	"o_calc_nox":		False,			  # legacy NOx lifetime fit
	"o_plot":			True,			  # produce PNG plots
	"o_plot_every":		1,			  # plot every N overpasses
	"o_plot_wddiff":	False,			  # overlay wind-direction comparison
	"snapshot":			True,			  # save code snapshot on completion

	# ------------------------------------------------------------------
	# Physical constants  (change only if using different conventions)
	# ------------------------------------------------------------------
	"EARTH_RADIUS_M":	6_371_000.0,	  # haversine Earth radius [m]
}

# Build target table once at import time
target = pd.read_csv(CFG["TARGET_INFO"])
target["facilityId"] = target["facilityId"].astype(str)
target = (target.loc[target["facilityId"].isin(CFG["TARGET_IDS_ALL"])]
				.loc[lambda d: d["facilityId"].isin(CFG["TARGET_IDS_RUN"])]
				.reset_index(drop=True))


# =============================================================================
# GEOMETRY HELPERS
# =============================================================================
def _haversine_m(lat1, lon1, lat2, lon2):
	"""Vectorised haversine distance [m]."""
	R	 = CFG["EARTH_RADIUS_M"]
	dlat = np.radians(lat2 - lat1)
	dlon = np.radians(lon2 - lon1)
	a	 = (np.sin(dlat / 2)**2
			+ np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2)
	return R * 2 * np.arcsin(np.sqrt(a))

def _calculate_bearing(lat1, lon1, lat2, lon2):
	"""Bearing from point-1 → point-2 [degrees, 0–360)."""
	lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
	y = np.sin(lon2 - lon1) * np.cos(lat2)
	x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
	return (np.degrees(np.arctan2(y, x)) + 360) % 360


# =============================================================================
# DATA READERS
# =============================================================================
def _read_tdump(satellite, tname_long, tid, time, time_utc_overpass):
	"""Read HYSPLIT tdump; return DataFrame or empty string on missing file."""
	d_tdump = CT.d_noxno2 + "d_dat/hysplit/" + CFG["HYSTRAJ_RUN_VER"] + "/" + tname_long + "/tdump/"
	matches = glob.glob(d_tdump + f"tdump_{satellite}_{tid}_*_{time}_{CFG['HYSTRAJ_RUN_VER']}")
	if not matches:
		with open("./csf_prcs.log", "a") as log:
			log.write(f"[tdump not found] {d_tdump}\n")
		return ""
	hdrs = ["?1","?2","year","month","day","hour","minute","forecast_hour",
			"age_hours","latitude","longitude","z_magl","pres","temp"]
	with open(matches[0]) as fh:
		nmet_ctr = int(fh.readline().strip().split()[0])
	tdump = pd.read_csv(matches[0], skiprows=4 + nmet_ctr,
						names=hdrs, sep=r"\s+", engine="python").reset_index(drop=True)
	tdump["tdump_id"] = np.arange(len(tdump))
	tdump["tstmp"] = [pd.to_datetime(f"20{tdump.loc[i,'year']}-{tdump.loc[i,'month']:02d}-"
									 f"{tdump.loc[i,'day']:02d} {tdump.loc[i,'hour']:02d}:"
									 f"{tdump.loc[i,'minute']:02d}") for i in range(len(tdump))]
	lats, lons = tdump["latitude"].values, tdump["longitude"].values
	step_m		= np.zeros(len(lats))
	step_m[1:]	= _haversine_m(lats[:-1], lons[:-1], lats[1:], lons[1:])
	tdump["age_km"] = np.cumsum(step_m) / 1000.0
	bearings	= _calculate_bearing(lats[0], lons[0], lats, lons)
	bearings[0] = bearings[1]
	tdump["wd"] = bearings
	dt = (tdump["tstmp"][2] - tdump["tstmp"][1]).seconds
	wso = _haversine_m(lats[:-1], lons[:-1], lats[1:],	lons[1:]) / dt
	u	= _haversine_m(lats[:-1], lons[:-1], lats[:-1], lons[1:]) / dt
	v	= _haversine_m(lats[:-1], lons[1:],  lats[1:],	lons[1:]) / dt
	tdump["wso"] = np.append(wso, -9999.)
	tdump["u"]	 = np.append(u,   -9999.)
	tdump["v"]	 = np.append(v,   -9999.)

	"""Add 'time_tag' (tz-naive UTC) to each trajectory row."""
	t0 = pd.Timestamp(time_utc_overpass)
	if t0.tzinfo is not None:
		t0 = t0.tz_convert("UTC").tz_localize(None)
	tdump["time_tag"] = t0 - (tdump["tstmp"] - tdump.loc[0, "tstmp"])

	return tdump


def _read_trop(dfin, tname_long, sate_row):
	"""Read TROPOMI NetCDF; return (Geo)DataFrame of QA-filtered pixels."""
	var_list = ["nitrogendioxide_tropospheric_column",
				"nitrogendioxide_tropospheric_column_precision",
				"qa_value", "time_utc", "latitude", "longitude"]
	if CFG["o_plot"]:
		var_list += ["longitude_bounds", "latitude_bounds"]
	with xr.open_dataset(dfin, engine="netcdf4", mask_and_scale=False) as ds:
		data = ds[var_list].load()
		scanline_idx	= data["scanline"].values		# shape (scanline,)
		groundpx_idx	= data["ground_pixel"].values	# shape (ground_pixel,)
	no2		 = data["nitrogendioxide_tropospheric_column"].values
	no2_prec = data["nitrogendioxide_tropospheric_column_precision"].values
	qa		 = data["qa_value"].values
	time_utc = data["time_utc"].values
	lat		 = data["latitude"].values
	lon		 = data["longitude"].values
	time_utc = np.repeat(time_utc[:, :, np.newaxis], no2.shape[2], axis=2)

	# build full 2-D index arrays matching (time, scanline, ground_pixel)
	# scanline varies along axis-1, ground_pixel along axis-2
	sl_2d, gp_2d = np.meshgrid(scanline_idx, groundpx_idx, indexing="ij")  # (scanline, ground_pixel)
	sl_3d = sl_2d[np.newaxis, :, :]   # broadcast over time dim
	gp_3d = gp_2d[np.newaxis, :, :]

	mask = (no2.ravel() >= 0.0) & (qa.ravel() >= CFG["QA_MIN"])
	trop = pd.DataFrame({
		"no2":			 no2.ravel()[mask] * 1e6,	# mol m-2 → µmol m-2
		"no2_precision": no2_prec.ravel()[mask],
		"qa_value":		 qa.ravel()[mask],
		"time_utc":		 time_utc.ravel()[mask],
		"latitude":		 lat.ravel()[mask],
		"longitude":	 lon.ravel()[mask],
		"scanline":		 sl_3d.ravel()[mask],
		"ground_pixel":  gp_3d.ravel()[mask],
	})
	if CFG["o_plot"]:
		lonb  = data["longitude_bounds"].values.reshape(-1, 4)[mask]
		latb  = data["latitude_bounds"].values.reshape(-1, 4)[mask]
		trop  = gpd.GeoDataFrame(trop, crs="EPSG:4326",
								 geometry=gpd.points_from_xy(trop.longitude, trop.latitude))
		finite = np.isfinite(lonb).all(axis=1) & np.isfinite(latb).all(axis=1)
		coords	= np.stack([lonb, latb], axis=2)
		closed	= np.concatenate([coords, coords[:, :1, :]], axis=1)
		polys	= np.where(finite, [Polygon(c) for c in closed], None)
		trop	= trop.assign(geom_poly=polys).set_geometry("geom_poly", crs="EPSG:4326")
	return trop

def _read_cems_nox(tid, tname, time_utc):
	"""Read CEMS hourly NOx for the lookback window ending at time_utc."""
	pattern = os.path.join(CT.d_cems + tname, f"{tname}_*.csv")
	files	= sorted(glob.glob(pattern))
	if not files:
		print(f"  [CEMS] No files: {pattern}")
		return pd.DataFrame()
	cems = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
	cems["tstmp_local"] = pd.to_datetime(cems["date"]) + pd.to_timedelta(cems["hour"], unit="h")
	row = pd.read_csv(CFG["TARGET_INFO"])
	row = row.loc[row["facilityId"].astype(str) == str(tid)].iloc[0]
	tz	= TimezoneFinder().timezone_at(lat=row["lat"], lng=row["lon"]) or "UTC"
	cems["tstmp_utc"] = (cems["tstmp_local"]
						 .dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
						 .dt.tz_convert("UTC").dt.tz_localize(None))
	t_end	= pd.Timestamp(time_utc)
	if t_end.tzinfo is not None:
		t_end = t_end.tz_convert("UTC").tz_localize(None)
	t_start = t_end - pd.Timedelta(hours=CFG["CEMS_LOOKBACK_HRS"])
	cems	= cems[(cems["tstmp_utc"] >= t_start) & (cems["tstmp_utc"] <= t_end)].copy()
	if cems.empty:
		print(f"  [CEMS] No data in {t_start} – {t_end} for {tid}")
		return pd.DataFrame()
	cems["noxMass"] = pd.to_numeric(cems["noxMass"], errors="coerce")
	hourly = (cems.groupby("tstmp_utc", as_index=False)["noxMass"].sum().rename(columns={"noxMass": "noxMass_tph"}))
	hourly["noxMass_tph_metric"] = hourly["noxMass_tph"] * 0.000453592	 # lbs/hr → t/hr
	return hourly


# =============================================================================
# SAMPLING & GAUSSIAN CONSTRAINTS
# =============================================================================
def _sample_soundings_along_traj(satellite, data, tdump):
	"""Sample satellite pixels onto orthogonal transects; return DataFrame."""
	val_col		= "no2" if satellite == "trop" else None
	dist_thresh = CFG["DIST_THRESH_TROP"]
	min_pix		= CFG["MIN_PIXELS_PER_TRANSECT"]
	dsec		= (tdump.loc[1, "tstmp"] - tdump.loc[0, "tstmp"]).seconds
	data_lon	= data["longitude"].values
	data_lat	= data["latitude"].values
	data_val	= data[val_col].values
	lons, lats	= tdump["longitude"].values, tdump["latitude"].values
	results		= []
	for i in range(len(tdump) - 1):
		lon  = (lons[i] + lons[i+1]) / 2
		lat  = (lats[i] + lats[i+1]) / 2
		dlon = lons[i+1] - lons[i]
		dlat = lats[i+1] - lats[i]
		hl_km = CFG["TRANSECT_HALFLENGTH"]						# [km]
		# bearing of trajectory segment → perpendicular bearing
		seg_bearing  = _calculate_bearing(lats[i], lons[i], lats[i+1], lons[i+1])
		perp_bearing = (seg_bearing + 90.0) % 360.0
		# sample transect at 0.5 km spacing in physical distance
		n_pts  = int(2 * hl_km / 0.5) + 1
		d_vals = np.linspace(-hl_km, hl_km, n_pts)				# signed km from centre
		R_km   = CFG["EARTH_RADIUS_M"] / 1e3
		d_rad  = d_vals / R_km
		lat_c, lon_c = np.radians(lat), np.radians(lon)
		pb_rad = np.radians(perp_bearing)
		lat_tmp = np.degrees(np.arcsin(
			np.sin(lat_c) * np.cos(d_rad) +
			np.cos(lat_c) * np.sin(d_rad) * np.cos(pb_rad)))
		lon_tmp = lon + np.degrees(np.arctan2(
			np.sin(pb_rad) * np.sin(d_rad) * np.cos(lat_c),
			np.cos(d_rad) - np.sin(lat_c) * np.sin(np.radians(lat_tmp))))
		mid		 = len(d_vals) // 2
		dist_tmp = d_vals							# already signed km
		dist_thresh_deg = dist_thresh / (np.pi / 180.0 * CFG["EARTH_RADIUS_M"] / 1e3)
		ddlon  = data_lon[np.newaxis, :] - lon_tmp[:, np.newaxis]
		ddlat  = data_lat[np.newaxis, :] - lat_tmp[:, np.newaxis]
		#nearby = (ddlon**2 + ddlat**2) < dist_thresh_deg**2
		#hit_rows = np.where(nearby.any(axis=1))[0]
		#if len(hit_rows) < min_pix:
		#	continue
		#rows = []
		#for j in hit_rows:
		#	idx = np.where(nearby[j])[0]
		#	rows.append({"lon_ortho":	lon_tmp[j],
		#				 "lat_ortho":	lat_tmp[j],
		#				 "dist_ortho":	dist_tmp[j],
		#				 val_col:		data_val[idx].mean(),
		#				 "tdump_id":	i,
		#				 "lon_sampled": data_lon[idx].mean(),
		#				 "lat_sampled": data_lat[idx].mean()})
		dist2_pix = ddlon**2 + ddlat**2			# (n_transect_pts, n_pixels)
		if CFG["SAMPLE_MODE"] == "nearest":
			# per-pixel: find the single closest transect point
			best_tp = np.argmin(dist2_pix, axis=0)	# (n_pixels,)
			in_band = dist2_pix[best_tp, np.arange(len(data_lon))] < dist_thresh_deg**2
			pix_idx = np.where(in_band)[0]			# pixels within threshold
			if len(pix_idx) < min_pix:
				continue
			rows = []
			for px in pix_idx:
				j = best_tp[px]
				rows.append({"lon_ortho":	lon_tmp[j],
							 "lat_ortho":	lat_tmp[j],
							 "dist_ortho":	dist_tmp[j],
							 val_col:		data_val[px],
							 "tdump_id":	i,
							 "lon_sampled": data_lon[px],
							 "lat_sampled": data_lat[px]})
		if CFG["SAMPLE_MODE"] == "mean_within_range":
			nearby	 = dist2_pix < dist_thresh_deg**2
			hit_rows = np.where(nearby.any(axis=1))[0]
			if len(hit_rows) < min_pix:
				continue
			rows = []
			for j in hit_rows:
				idx = np.where(nearby[j])[0]
				rows.append({"lon_ortho":	lon_tmp[j],
							 "lat_ortho":	lat_tmp[j],
							 "dist_ortho":	dist_tmp[j],
							 val_col:		data_val[idx].mean(),
							 "tdump_id":	i,
							 "lon_sampled": data_lon[idx].mean(),
							 "lat_sampled": data_lat[idx].mean()})

		if len(rows) >= min_pix:
			results.append(pd.DataFrame(rows))
	if not results:
		return pd.DataFrame()
	sampled = pd.concat(results, ignore_index=True)
	return sampled[sampled.groupby("tdump_id")["tdump_id"].transform("size") >= min_pix]

def _define_gaussian_constraints(satellite, data_sampled):
	"""Return per-transect Gaussian fitting bounds from CFG["GAUSS_BOUNDS"]."""
	b = CFG["GAUSS_BOUNDS"]
	c = pd.DataFrame({"tdump_id": data_sampled["tdump_id"].unique()})
	for k in range(5):
		name = f"a{k}"
		c[f"{name}_min"] = b[name][0]
		c[f"{name}_max"] = b[name][1]
	c["dtdump_id_min"] = 0.
	return c


# =============================================================================
# CSF CALCULATION
# =============================================================================
def _calc_csf(satellite, target, data, constraint, tdump, suffix):
	"""
	Fit Gaussian to each transect; return per-transect CSF metrics.

	Inputs (from caller)
	--------------------
	data	   : output of _sample_soundings_along_traj  — columns: tdump_id, dist_ortho, no2, lon_ortho, lat_ortho
	constraint : output of _define_gaussian_constraints  — columns: tdump_id, a0_min/max … a4_min/max
	tdump	   : HYSPLIT or optimised trajectory DataFrame — columns: tdump_id, longitude, latitude,
				 age_hours, month, hour, temp, pres, wso, wd, time_tag
	suffix	   : str, "_H" or "_O" — appended to every output column

	Columns added to output DataFrame
	-----------------------------------
	Join keys (no suffix — row identity, not measurements):
	  tdump_id

	Trajectory metadata (suffixed — source depends on which tdump was passed):
	  lon_tdump{s}, lat_tdump{s}, age_hours{s}, month{s}, hour{s},
	  temp{s}, pres{s}, wso{s}, wd{s}, time_tag{s}

	Gaussian fit parameters (suffixed):
	  a0{s}, a1{s}, a2{s}, a3{s}, a4{s}			  — best-fit coefficients
	  a0sig{s} … a4sig{s}						   — 1-sigma uncertainties
	  a0sigpct{s} … a4sigpct{s}					   — uncertainty as % of value
	  lon_a3{s}, lat_a3{s}						   — geographic location of Gaussian peak
	  lon_a3_i{s}, lat_a3_i{s}					   — location of left  FWHM edge (o_plot only)
	  lon_a3_f{s}, lat_a3_f{s}					   — location of right FWHM edge (o_plot only)

	Gaussian quality diagnostics (suffixed):
	  d_gap{s}, d_left{s}, d_right{s}, d_center{s} — sampling coverage metrics [%]
	  nobs{s}, nobs_left{s}, nobs_right{s}			— pixel counts
	  mae{s}, mape{s}, rsq{s}, rsq_detrend{s}		— fit quality scores
	  dtdump_id_min{s}								 — from constraint table

	Flux (suffixed, TROPOMI only):
	  flux_no2{s}	[tNO2/hr]

	Wind summary (suffixed, computed over all transects):
	  wd_std{s}		— circular std of wind direction [deg]
	  wd_diff{s}	— bearing difference: a3-chain vs tdump-chain [deg]
	"""
	s  = suffix   # short alias used throughout
	merge_cols = ["tdump_id","longitude","latitude","age_hours","month","hour","temp","pres","wso","wd","time_tag"]
	out = (pd.DataFrame({"tdump_id": data["tdump_id"].unique()})
			 .merge(tdump[merge_cols], on="tdump_id", how="left")
			 .rename(columns={
				 "longitude": f"lon_tdump{s}", "latitude":	f"lat_tdump{s}",
				 "age_hours": f"age_hours{s}", "month":		f"month{s}",
				 "hour":	  f"hour{s}",	   "temp":		f"temp{s}",
				 "pres":	  f"pres{s}",	   "wso":		f"wso{s}",
				 "wd":		  f"wd{s}",		   "time_tag":	f"time_tag{s}",
			 }))

	out[f"month{s}"] = out[f"month{s}"].values.astype(int)
	voi = "no2" if satellite == "trop" else None
	for i in range(len(out)):
		tmp = data.loc[data["tdump_id"] == out.loc[i, "tdump_id"]].reset_index(drop=True)
		if len(tmp) <= 1:
			continue
		bounds = ([constraint.loc[i, f"a{k}_min"] for k in range(5)],
				  [constraint.loc[i, f"a{k}_max"] for k in range(5)])
		out.loc[i, f"dtdump_id_min{s}"] = constraint.loc[i, "dtdump_id_min"]
		x, y   = tmp["dist_ortho"].values, tmp[voi].values
		popt, pcov = curve_fit(FN._gaussian, x, y, bounds=bounds, maxfev=CFG["MAXFEV"])
		a0, a1, a2, a3, a4 = popt
		perr = np.sqrt(np.diag(pcov))
		tmp["del_dist_ortho"] = np.abs(tmp["dist_ortho"].values - a3)
		out.loc[i, f"lon_a3{s}"] = tmp.loc[tmp["del_dist_ortho"].idxmin(), "lon_ortho"]
		out.loc[i, f"lat_a3{s}"] = tmp.loc[tmp["del_dist_ortho"].idxmin(), "lat_ortho"]
		if CFG["o_plot"]:
			for side, offset in [("i", -(a4/2)), ("f", +(a4/2))]:
				tmp[f"del_dist_{side}"] = np.abs(tmp["dist_ortho"].values - (a3 + offset))
				out.loc[i, f"lon_a3_{side}{s}"] = tmp.loc[tmp[f"del_dist_{side}"].idxmin(), "lon_ortho"]
				out.loc[i, f"lat_a3_{side}{s}"] = tmp.loc[tmp[f"del_dist_{side}"].idxmin(), "lat_ortho"]
		for k, name in enumerate(["a0","a1","a2","a3","a4"]):
			out.loc[i, name+s]			= popt[k]
			out.loc[i, name+"sig"+s]	= perr[k]
			out.loc[i, name+"sigpct"+s] = perr[k] / (popt[k] + 1e-9) * 100.
		sigma	   = a4 / np.sqrt(8. * np.log(2))
		width_2sig = sigma * 4.
		gaps	   = np.abs(np.diff(tmp["dist_ortho"].values))
		out.loc[i, f"d_gap{s}"]    = gaps.max() / width_2sig * 100.
		out.loc[i, f"d_right{s}"]  = (tmp["dist_ortho"].max() - (a3 + sigma*2)) / width_2sig * 100.
		out.loc[i, f"d_left{s}"]   = ((a3 - sigma*2) - tmp["dist_ortho"].min()) / width_2sig * 100.
		out.loc[i, f"d_center{s}"] = np.abs(tmp["dist_ortho"] - a3).min() / width_2sig * 100.
		out.loc[i, f"nobs{s}"]		 = len(tmp)
		out.loc[i, f"nobs_left{s}"]  = (tmp["dist_ortho"] < a3).sum()
		out.loc[i, f"nobs_right{s}"] = (tmp["dist_ortho"] > a3).sum()
		y_fit = FN._gaussian(tmp["dist_ortho"], *popt)
		out.loc[i, f"mae{s}"]  = np.abs(tmp[voi].values - y_fit).mean()
		out.loc[i, f"mape{s}"] = sklearn.metrics.mean_absolute_percentage_error(tmp[voi].values, y_fit)
		out.loc[i, f"rsq{s}"]  = sklearn.metrics.r2_score(tmp[voi].values, y_fit)
		trend = a0 + a1 * tmp["dist_ortho"].values
		out.loc[i, f"rsq_detrend{s}"] = sklearn.metrics.r2_score(
			tmp[voi].values - trend, y_fit - trend)
		if satellite == "trop":
			out.loc[i, f"flux_no2{s}"] = (
				0.5 * (np.pi / np.log(2))**0.5
				* (a2 / 1e6 * 46.0055 / 1e6)   # µmol m-2 → t m-2
				* (a4 * 1e3)					# km → m
				* out.loc[i, f"wso{s}"] * 3600.)  # m s-1 → m hr-1

	if f"flux_no2{s}" in out.columns or f"flux_co2{s}" in out.columns:
		wd_rad = np.radians(out[f"wd{s}"].values)
		wd_R   = np.sqrt(np.nanmean(np.sin(wd_rad))**2 + np.nanmean(np.cos(wd_rad))**2)
		out[f"wd_std{s}"] = np.degrees(np.sqrt(-2. * np.log(wd_R)))

		out[f"wd_diff{s}"] = None
		if f"lat_a3{s}" in out.columns and len(out) > 1:
			wd_a3  = _calculate_bearing(out[f"lat_a3{s}"].iloc[0],	 out[f"lon_a3{s}"].iloc[0],
										out[f"lat_a3{s}"].iloc[1:].to_numpy(), out[f"lon_a3{s}"].iloc[1:].to_numpy())
			wd_traj = _calculate_bearing(out[f"lat_tdump{s}"].iloc[0], out[f"lon_tdump{s}"].iloc[0],
										 out[f"lat_tdump{s}"].iloc[1:].to_numpy(), out[f"lon_tdump{s}"].iloc[1:].to_numpy())
			wd_diff = (np.asarray(wd_a3) - np.asarray(wd_traj) + 180.) % 360. - 180.
			out[f"wd_diff{s}"] = np.insert(wd_diff, 0, wd_diff[0])

	return out


# =============================================================================
# GAUSSIAN QUALITY FLAG
# =============================================================================
def _calc_qf_gauss(csf, suffix=""):
	"""
	Add QF_gauss_abs{suffix} column: 0 = good, 1 = bad.
	Thresholds are read from CFG["QF"].
	"""
	q	= CFG["QF"]
	s	= suffix
	out = csf.copy()

	def col(name):
		return out[name + s] if (name + s) in out.columns else pd.Series(np.nan, index=out.index)

	good = (
		(col("rsq_detrend") >= q["rsq_detrend_min"])  &
		(col("d_gap")		<= q["d_gap_max"])		   &
		(col("d_left")		>= q["d_left_min"])		   &
		(col("d_right")		>= q["d_right_min"])	   &
		(col("d_center")	<= q["d_center_max"])	   &
		(col("a2sigpct")	<= q["a2sigpct_max"])	   &
		(col("a4sigpct")	<= q["a4sigpct_max"])	   &
		(col("nobs")		>= q["nobs_min"])
	)
	out[f"QF_gauss_abs{s}"] = np.where(good, 0, 1)
	return out


# =============================================================================
# TRAJECTORY HELPERS
# =============================================================================
def _TDOPT_a3(data_csf, tdump, suffix):
	"""Build optimised trajectory from Gaussian peak locations (a3 lat/lon)."""
	valid = data_csf.dropna(subset=[f"lon_a3{suffix}", f"lat_a3{suffix}"]).reset_index(drop=True)
	if len(valid) < 2:
		return pd.DataFrame()
	base = tdump.loc[tdump["tdump_id"].isin(valid["tdump_id"])].reset_index(drop=True)
	true = pd.DataFrame({
		"tdump_id":  valid["tdump_id"].values,
		"longitude": valid[f"lon_a3{suffix}"].astype(float).values,
		"latitude":  valid[f"lat_a3{suffix}"].astype(float).values,
		"age_hours": valid[f"age_hours{suffix}"].values,
		"month":	 valid[f"month{suffix}"].values,
		"hour":		 valid[f"hour{suffix}"].values,
		"temp":		 valid[f"temp{suffix}"].values,
		"pres":		 valid[f"pres{suffix}"].values,
		"tstmp":	 base["tstmp"].values,
		"time_tag":  base["time_tag"].values,
	}).reset_index(drop=True)
	lats, lons = true["latitude"].values, true["longitude"].values
	dt	= (tdump.loc[1,"tstmp"] - tdump.loc[0,"tstmp"]).seconds
	wso = _haversine_m(lats[:-1], lons[:-1], lats[1:],	lons[1:]) / dt
	u	= _haversine_m(lats[:-1], lons[:-1], lats[:-1], lons[1:]) / dt
	v	= _haversine_m(lats[:-1], lons[1:],  lats[1:],	lons[1:]) / dt
	true["wso"] = np.append(wso, wso[-1])
	true["u"]	= np.append(u,	 u[-1])
	true["v"]	= np.append(v,	 v[-1])
	bearings	= _calculate_bearing(lats[0], lons[0], lats, lons)
	bearings[0] = bearings[1]
	true["wd"]	= bearings
	return true


# =============================================================================
# HIGH-NO2 PIXEL-CHAIN TRAJECTORY
# =============================================================================
# Add to CFG["HIGHNO2"] block:
#	"search_radius_deg": 0.15,	 # neighbor search radius [deg] — ~2–3 TROPOMI pixels
#	"bg_snr_thresh":	 1.0,	 # stop when local max < zbg + thresh * zbg_sig
#	"max_steps":		 60,	 # hard ceiling on walk length
# CFG addition:
# "HIGHNO2": {
#	  "bg_snr_thresh": 1.0,   # stop when cell no2 < zbg + thresh * zbg_sig
#	  "max_steps":	   60,	  # hard ceiling on chain length
# },

def _TDOPT_moore(trop, target_row, tdump):
	"""
	Build a plume-following trajectory by chaining peak-NO2 TROPOMI pixels
	downwind from the source, using a Moore (3x3) neighbourhood walk on
	the native scanline × ground_pixel raster.

	Steps
	-----
	1. Build a lookup dict (scanline, ground_pixel) -> no2 from trop.
	2. Compute scene background zbg (median) and zbg_sig (IQR).
	3. Seed at the pixel nearest the facility (lon/lat distance).
	4. Walk: at each step move to the unvisited 3x3 Moore neighbour
	   with the highest NO2 still above (zbg + bg_snr_thresh * zbg_sig).
	5. Map each waypoint to the nearest tdump row to inherit meteorology.

	Returns
	-------
	DataFrame with same schema as _build_true_tdump output,
	or empty DataFrame if fewer than 2 waypoints were found.
	"""
	cfg_h  = CFG['TDOPT']
	thresh = cfg_h["bg_snr_thresh"]
	maxstp = cfg_h["max_steps"]

	sls  = trop["scanline"].values
	gps  = trop["ground_pixel"].values
	lons = trop["longitude"].values
	lats = trop["latitude"].values
	no2  = trop["no2"].values

	# ------------------------------------------------------------------ #
	# 1. Build raster lookup on native TROPOMI grid indices				  #
	# ------------------------------------------------------------------ #
	grid	 = {}	# (sl, gp) -> no2
	cell_lon = {}	# (sl, gp) -> longitude
	cell_lat = {}	# (sl, gp) -> latitude
	for k in range(len(no2)):
		key = (int(sls[k]), int(gps[k]))
		grid[key]	  = no2[k]
		cell_lon[key] = lons[k]
		cell_lat[key] = lats[k]

	# ------------------------------------------------------------------ #
	# 2. Background stats (scene-wide)									  #
	# ------------------------------------------------------------------ #
	no2_vals	= np.array(list(grid.values()))
	zbg			= np.nanmedian(no2_vals)
	zbg_sig		= np.nanquantile(no2_vals, 0.75) - np.nanquantile(no2_vals, 0.25)
	plume_floor = zbg + thresh * zbg_sig

	# ------------------------------------------------------------------ #
	# 3. Seed: pixel with minimum lon/lat distance to facility			  #
	# ------------------------------------------------------------------ #
	dist2 = (lons - target_row["lon"])**2 + (lats - target_row["lat"])**2
	k0	  = int(np.argmin(dist2))
	seed  = (int(sls[k0]), int(gps[k0]))

	# ------------------------------------------------------------------ #
	# 4. Moore-neighbourhood walk on (scanline, ground_pixel) grid		  #
	# ------------------------------------------------------------------ #
	visited   = set()
	waypoints = []		# list of (sl, gp) keys

	cur = seed
	for _ in range(maxstp):
		visited.add(cur)
		waypoints.append(cur)

		best_key = None
		best_no2 = -np.inf
		for dsl in range(-1, 2):
			for dgp in range(-1, 2):
				if dsl == 0 and dgp == 0:
					continue
				nb = (cur[0] + dsl, cur[1] + dgp)
				if nb in visited or nb not in grid:
					continue
				if grid[nb] >= plume_floor and grid[nb] > best_no2:
					best_no2 = grid[nb]
					best_key = nb

		if best_key is None:
			break

		cur = best_key

	if len(waypoints) < 2:
		return pd.DataFrame()

	# ------------------------------------------------------------------ #
	# 5. Match waypoints to nearest tdump row; assemble output			  #
	# ------------------------------------------------------------------ #
	tdump_lats = tdump["latitude"].values
	tdump_lons = tdump["longitude"].values

	rows = []
	for key in waypoints:
		wp_lon = cell_lon[key]
		wp_lat = cell_lat[key]
		d_m    = _haversine_m(wp_lat, wp_lon, tdump_lats, tdump_lons)
		j	   = int(np.argmin(d_m))
		td	   = tdump.iloc[j]
		rows.append({
			"tdump_id":  int(td["tdump_id"]),
			"longitude": wp_lon,
			"latitude":  wp_lat,
			"no2":		 grid[key],
			"age_hours": td["age_hours"],
			"age_km":	 td["age_km"],
			"month":	 int(td["month"]),
			"hour":		 int(td["hour"]),
			"temp":		 td["temp"],
			"pres":		 td["pres"],
			"tstmp":	 td["tstmp"],
			"time_tag":  td["time_tag"],
			"wso":		 td["wso"],
		})

	chain = pd.DataFrame(rows).reset_index(drop=True)

	# recompute wind direction from chained geometry
	clats = chain["latitude"].values
	clons = chain["longitude"].values
	dt	  = (tdump.loc[1, "tstmp"] - tdump.loc[0, "tstmp"]).seconds
	bearings	 = _calculate_bearing(clats[0], clons[0], clats, clons)
	bearings[0]  = bearings[1]
	chain["wd"]  = bearings

	# wind speed summary: mean ± std across HYSPLIT-matched rows
	# stored as scalars broadcast to all rows — consistent with how
	# _calc_csf consumes wso (per-row, but nearly constant along a short plume)
	chain["wso__std"] = chain['wso'].std()
	chain["wso"] = chain['wso'].mean()

	chain = chain.sort_values(['age_km']).reset_index(drop=True)
	chain['tdump_id']= chain.index

	return chain

def _smooth_chain_trajectory(chain, tdump, poly_deg=2):
	"""
	Fit a polynomial lon(age_km), lat(age_km) through the raw
	tdump_highNO2 chain to remove zigzag noise.
	Returns a tdump-shaped DataFrame resampled at original chain spacing.
	"""
	chain=	chain.drop_duplicates("age_km").reset_index(drop=True)

	if len(chain) < poly_deg + 1:
		return chain

	t	= np.unique(chain["age_km"].values)
	t_n = (t - t[0]) / (t[-1] - t[0] + 1e-9) # normalize t to [0,1] for numerical stability

	plon = np.polyfit(t_n, chain["longitude"].values, poly_deg)
	plat = np.polyfit(t_n, chain["latitude"].values,  poly_deg)

	smoothed = chain.copy()
	smoothed["longitude"] = np.polyval(plon, t_n)
	smoothed["latitude"]  = np.polyval(plat, t_n)

	# recompute wind fields on smoothed geometry
	clats = smoothed["latitude"].values
	clons = smoothed["longitude"].values
	dt	  = (tdump.loc[1, "tstmp"] - tdump.loc[0, "tstmp"]).seconds
	wso   = _haversine_m(clats[:-1], clons[:-1], clats[1:],  clons[1:])  / dt
	u	  = _haversine_m(clats[:-1], clons[:-1], clats[:-1], clons[1:])  / dt
	v	  = _haversine_m(clats[:-1], clons[1:],  clats[1:],  clons[1:])  / dt
	bearings		= _calculate_bearing(clats[0], clons[0], clats, clons)
	bearings[0]		= bearings[1]
	smoothed["wd"]	= bearings
	smoothed['tdump_id']=	smoothed.index.values

	return smoothed




# =============================================================================
# OUTPUT HELPERS
# =============================================================================
def _build_output_dirs(dout_csv, dout_png):
	os.makedirs(dout_csv, exist_ok=True)
	os.makedirs(dout_png, exist_ok=True)

def _collect_satellite_files(tname):
	"""Return date-filtered DataFrame of TROPOMI overpasses for tname."""
	csv_path = os.path.join(CT.d_noxno2, "d_dat", "trop", CFG["SATE_INFO"][2],
							"d_csv", f"{tname}_{CFG['SATE_INFO'][2]}.csv")
	if not os.path.isfile(csv_path):
		raise FileNotFoundError(f"[_collect_satellite_files] {csv_path}")
	sate = pd.read_csv(csv_path)
	sate["time_utc_i"] = pd.to_datetime(sate["time_utc_i"], utc=True).dt.tz_localize(None)
	sate["time_utc_f"] = pd.to_datetime(sate["time_utc_f"], utc=True).dt.tz_localize(None)
	sate["time_utc"]   = sate["time_utc_i"]
	d_nc = CT.d_trop_target + CFG["SATE_INFO"][2] + "/d_nc/" + tname + "/"
	sate["dfin"] = d_nc + sate["file"] + "_" + tname + "_" + CFG["SATE_INFO"][2] + ".nc"
	sate = sate[(sate["time_utc"] >= pd.Timestamp(CFG["SATE_INFO"][3])) &
				(sate["time_utc"] <= pd.Timestamp(CFG["SATE_INFO"][4]))].reset_index(drop=True)
	return sate


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================
def _csf_prcs(CFG, target):
	"""
	For each target × overpass:
	  2a. Read HYSPLIT trajectory
	  2b. Read TROPOMI pixels
	  2c. Tag trajectory with backward UTC times
	  2d. Sample pixels along HYSPLIT trajectory
	  2e. First-pass Gaussian CSF  (_H suffix) + QF
		___ 2f - 2h are optional ___
	  2f. Build optimised trajectory from a3 peak locations
	  2g. Re-sample on optimised trajectory
	  2h. Optimised Gaussian CSF  (_O suffix) + QF
		____________________________
	  2i. Merge, save CSV, run NOx workflow
	  2j. CEMS lookup
	  2k. Plot
	"""
	for it in range(len(target)):
		tid   = target.loc[it, "facilityId"]
		tname = (tid + "_" + target.loc[it, "facilityName"]).replace(" ", "_")
		dout_csv = CT.d_noxno2 + "d_dat/csf/" + CFG["CSF_PRCS_VER"] + "/" + tname + "/"
		dout_png = CT.d_noxno2 + "d_fig/csf/" + CFG["CSF_PRCS_VER"] + "/" + tname + "/"
		_build_output_dirs(dout_csv, dout_png)
		print(f"Processing {it}/{len(target)-1}: {tid} {target.loc[it,'facilityName']}")

		sate = _collect_satellite_files(tname)

		for i_sate in range(len(sate)):
			sate_row = sate.loc[i_sate]
			tstr	 = sate_row["time_utc"].strftime("%Y%m%d%H%M")
			fout	 = f"csf_{sate_row['file']}_{CFG['CSF_PRCS_VER']}"
			print(f"  [{it}/{len(target)-1}] {tid}	overpass {i_sate}/{len(sate)-1}  {tstr}")

			# a. Read HYSPLIT trajectory
			tdump = _read_tdump("trop", tname, tid, tstr, sate_row["time_utc"])
			if len(tdump) <= 1:
				continue

			# b. Read TROPOMI
			trop = _read_trop(sate_row["dfin"], tname, sate_row)

			# c. Sample satellite pixels along cross-sections of HYSPLIT trajectory
			trop_sampled = _sample_soundings_along_traj("trop", trop, tdump)
			if len(trop_sampled) == 0:
				continue

			# d. Calculate CSF (HYSPLIT-based, _H) + quality flag
			constraint = _define_gaussian_constraints("trop", trop_sampled)
			trop_csf   = _calc_csf("trop", target.loc[it], trop_sampled, constraint, tdump, suffix="_H")
			trop_csf   = _calc_qf_gauss(trop_csf, suffix="_H")

			# e. Calculate CSF usint optimized trajectory (_O) + quality flag
			if CFG['TDOPT']['method'] == None:
				true_tdump = tdump.copy()
			else:
				# Build optimised trajectory using a3
				if CFG['TDOPT']['method'] == 'a3':
					true_tdump = _TDOPT_a3(trop_csf, tdump, suffix="_H")

				# Build optimised trajectory using moore
				if CFG['TDOPT']['method'] == 'moore':
					true_tdump = _TDOPT_moore(trop, target.loc[it], tdump)
					true_tdump = _smooth_chain_trajectory(true_tdump, tdump, poly_deg=CFG["TDOPT"]["smooth_poly_deg"])

				# Calculated optimised CSF (_O) + quality flag
				trop_true_sampled = pd.DataFrame()
				if len(true_tdump) >= 2:
					trop_true_sampled = _sample_soundings_along_traj("trop", trop, true_tdump)
					if len(trop_true_sampled) > 0:
						constraint_true = _define_gaussian_constraints("trop", trop_true_sampled)
						csf_true = _calc_csf("trop", target.loc[it], trop_true_sampled, constraint_true, true_tdump, suffix="_O")
						csf_true = _calc_qf_gauss(csf_true, suffix="_O")
						trop_csf = trop_csf.merge(csf_true, on="tdump_id", how="left")
					else:
						continue
				else:
					continue

			# 2f. Convert NO2 to NOx CSF, then save csv
			IPython.embed()

			trop_csf, nox_result = run_nox_workflow(
				trop_csf	  = trop_csf,
				true_tdump	  = true_tdump,
				d_geoscf	  = CT.d_geoscf,
				suffix		  = CFG["NOX_SUFFIX"],
				option_b_or_a = CFG["NOX_OPTION"],
				residual_tol  = CFG["NOX_RESIDUAL_TOL"],
			)
			trop_csf.to_csv(dout_csv + fout + ".csv", index=False)

			# 2g. Read CEMS, then plot
			if (CFG["o_plot"] and i_sate % CFG["o_plot_every"] == 0 and len(trop_csf) > 1):
				cems_nox = _read_cems_nox(tid, tname, sate_row["time_utc"])
				if not cems_nox.empty:
					print(f"  [CEMS] {len(cems_nox)} records, " f"mean = {cems_nox['noxMass_tph_metric'].mean():.2f} {CFG['FLUX_UNIT']}")

				plot_kwargs = dict(
					CFG				 = CFG,
					target			 = target.loc[it],
					data			 = trop,
					data_sampled	 = trop_sampled,
					data_csf		 = trop_csf,
					tdump			 = tdump,
					sate			 = sate_row,
					flux_unit		 = CFG["FLUX_UNIT"],
					o_plot_wddiff	 = CFG["o_plot_wddiff"],
					dfout			 = dout_png + fout,
					cems_nox		 = cems_nox,
					true_tdump		 = true_tdump,
					nox_result		 = nox_result,
				)

				if CFG['TDOPT']['method'] != None:
					plot_kwargs['data_true_sampled'] = trop_true_sampled

				_plot_csf(**plot_kwargs)

	#return trop_csf


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
	_ = _csf_prcs(CFG, target)
	if CFG["snapshot"]:
		FN._SNAPSHOT(name="csf_prcs", tag=CFG["CSF_PRCS_VER"],
					 scripts=["csf_prcs.py"], cfg=CFG,
					 out_dir=os.path.join(CT.d_noxno2, "d_history"))

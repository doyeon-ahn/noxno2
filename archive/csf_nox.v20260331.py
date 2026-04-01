# =============================================================================
# csf_geoscf_nox.py  —	GEOS-CF-driven NOx emission workflow
# Drop-in companion to csf_prcs.py (v20260324)
# =============================================================================
# Implements the five-step PSS back-calculation workflow using GEOS-CF fields:
#	Step 1 — PSS ratio along trajectory		  (j_NO2, O3, T from GEOS-CF)
#	Step 2 — Identify the PSS point			  (quality screening + flux peak)
#	Step 3 — NO2 flux → NOx flux at PSS
#	Step 4 — Decay correction back to t=0
#	Step 5 — Compare flux_nox_source to CEMS NOx emission rate
#
# GEOS-CF collections used
# -------------------------
#	chm_tavg_1hr_glo_L1440x721_slv	→  O3  [mol mol-1]
#	met_tavg_1hr_glo_L1440x721_slv	→  T   [K]
#	xgc_tavg_1hr_glo_L1440x721_slv	→  j_NO2 is NOT a direct GEOS-CF output;
#		we reconstruct it from COSZ (xgc collection) using the standard
#		Madronich (1987) two-stream parameterisation.
#	oxi_inst_1hr_glo_L1440x721_v72	→  OH  [mol mol-1]	(for Option B k_loss)
#
# All GEOS-CF files are expected to follow the v2 file-naming convention:
#	GEOS.cf.ana.<collection>.<yyyyMMdd_hhmmz>.RX.nc4	# R1: the first revision of the file; Reads whatever revision version exist in dir
# and be reachable via the NCCS OPeNDAP or local mirror path CT.d_geoscf.
# =============================================================================

import os, IPython, glob
import math
import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import curve_fit
from scipy.stats import linregress
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# Assumed import of project constants (same as csf_prcs.py)
# ---------------------------------------------------------------------------
# import CT			  # must define CT.d_geoscf (root of GEOS-CF file tree)
# from csf_prcs import _haversine_m   # re-use existing helper

# ---------------------------------------------------------------------------
# Physical / chemical constants
# ---------------------------------------------------------------------------
# Boltzmann rate constant for NO + O3 → NO2 + O2  [cm3 molec-1 s-1]
# Arrhenius form: k = A * exp(-Ea/RT) with R = 8.314 J mol-1 K-1
_A_NO_O3   = 3.0e-12	   # [cm3 molec-1 s-1]	 pre-exponential factor
_Ea_NO_O3  = 1500.0		   # [K]				  activation temperature

# Avogadro's number
_NA = 6.02214076e23		   # [molec mol-1]

# Mean sea-level pressure for number density calculation
_P0_PA = 101325.0		   # [Pa]

# Molecular weight of NO2 (g mol-1)
_MW_NO2 = 46.0055

# Reference J_NO2 at SZA = 0  [s-1]  (clear-sky, overhead sun)
# Sander et al. (2011) recommend ~8.5e-3 s-1 for boundary layer
_JNO2_OVERHEAD = 8.5e-3    # [s-1]


# =============================================================================
# GEOS-CF FILE HELPERS
# =============================================================================
def _geoscf_filepath(collection, timestamp_utc, d_geoscf, mode="ana"):
	"""
	Build the full path to a GEOS-CF v2 NetCDF-4 file.
	Parameters
	----------
	collection	  : str   e.g. "chm_tavg_1hr_glo_L1440x721_slv"
	timestamp_utc : pd.Timestamp  (tz-naive UTC)
	d_geoscf	  : str   root directory, e.g. CT.d_geoscf
	mode		  : str   "ana" or "fcst"
	Returns
	-------
	str  full file path
	"""
	# Round DOWN to nearest collection hour (tavg files are stamped at hh:30, inst files at hh:00; we just match the hour for look-up)
	ts	  = pd.Timestamp(timestamp_utc)
	hh	  = ts.floor("h").strftime("%H")
	mm	  = "30" if "tavg" in collection else "00"
	stamp = f"{ts.strftime('%Y%m%d')}_{hh}{mm}z"

	# Select directory tree and filename format based on data period:
	#	Jan 2023 – Oct 2024  → hindcast archive
	#	Aug 2025 – present	 → NRTv2
	#	anything in between  → not available, return ""
	_HINDCAST_START = pd.Timestamp("2023-01-01")
	_HINDCAST_END	= pd.Timestamp("2024-10-31 23:59:59")
	_NRT_START		= pd.Timestamp("2025-08-01")

	ymd = f"Y{ts.strftime('%Y')}/M{ts.strftime('%m')}/D{ts.strftime('%d')}"

	if _HINDCAST_START <= ts <= _HINDCAST_END:
		# e.g. CF2_hindcast.chm_tavg_1hr_glo_L1440x721_slv.20230101_1030z.R0.nc4
		fname = f"CF2_hindcast.{collection}.{stamp}*nc4"
		ddir  = os.path.join(d_geoscf, "hindcast", "pub", mode, ymd)
	elif ts >= _NRT_START:
		# e.g. GEOS.cf.ana.chm_tavg_1hr_glo_L1440x721_slv.20250801_1030z.R0.nc4
		fname = f"GEOS.cf.{mode}.{collection}.{stamp}*nc4"
		ddir  = os.path.join(d_geoscf, "NRTv2", "pub", mode, ymd)
	else:
		return ""  # gap period: no data available

	df = glob.glob(os.path.join(ddir, fname))
	return df[0] if df else ""


def _geoscf_interp_point(fpath, varname, lat, lon, lev_idx=0):
	"""
	Bilinearly interpolate a GEOS-CF surface/single-level variable to a
	given (lat, lon) point.

	Parameters
	----------
	fpath	: str	   path to GEOS-CF NetCDF-4 file
	varname : str	   variable name inside the file
	lat		: float    target latitude	[degrees_north]
	lon		: float    target longitude [degrees_east]
	lev_idx : int	   vertical level index (0 = surface / bottom layer 72)

	Returns
	-------
	float  interpolated value (scalar), or np.nan on failure
	"""
	if not os.path.isfile(fpath):
		return np.nan
	with xr.open_dataset(fpath, engine="netcdf4") as ds:
		if varname not in ds:
			return np.nan
		da = ds[varname]
		# Select time = 0 (one timestamp per granule)
		da = da.isel(time=0)
		# If 4-D (lev dimension present), select the requested level
		if "lev" in da.dims:
			da = da.isel(lev=lev_idx)
		# xarray nearest-neighbour then bilinear via interp
		val = float(
			da.interp(lat=lat, lon=lon, method="linear",
					  kwargs={"fill_value": "extrapolate"}).values
		)
	return val


def _geoscf_interp_column(fpath, varname, lat, lon):
	"""
	Return the full vertical profile (all model levels) interpolated to
	(lat, lon).  Used for OH column look-up.

	Returns
	-------
	np.ndarray shape (72,) or None on failure
	"""
	if not os.path.isfile(fpath):
		return None
	with xr.open_dataset(fpath, engine="netcdf4") as ds:
		if varname not in ds:
			return None
		da = ds[varname].isel(time=0)
		col = da.interp(lat=lat, lon=lon, method="linear",
						kwargs={"fill_value": "extrapolate"}).values
	return col.astype(float)


# =============================================================================
# STEP 1 — PSS ratio along trajectory
#			Requires: j_NO2, [O3], T  (all from GEOS-CF)
# =============================================================================

def _jno2_from_cosz(cosz):
	"""
	Estimate J_NO2 [s-1] from cosine of solar zenith angle.
	Uses a simple two-stream parameterisation following
	Madronich (1987) and Rohrer & Berresheim (2006):
		J_NO2 = J_NO2_overhead * max(0, cos(SZA))^p
	with p ≈ 0.7 (empirical, boundary layer, clear sky).

	For cloudy or twilight conditions this is approximate;
	for higher accuracy replace with a look-up table or TUV output.

	Parameters
	----------
	cosz : float  cosine of solar zenith angle (from GEOS-CF COSZ)

	Returns
	-------
	float  J_NO2 [s-1]
	"""
	cosz = max(0.0, float(cosz))
	return _JNO2_OVERHEAD * (cosz ** 0.7)


def _k_no_o3(T_K):
	"""
	Temperature-dependent bimolecular rate constant for
	NO + O3 → NO2 + O2	[cm3 molec-1 s-1]
	JPL Publication 19-5 (2019) recommendation.

	Parameters
	----------
	T_K : float  temperature [K]

	Returns
	-------
	float  k  [cm3 molec-1 s-1]
	"""
	return _A_NO_O3 * np.exp(-_Ea_NO_O3 / T_K)


def _o3_number_density(o3_molmol, T_K, P_Pa):
	"""
	Convert O3 mole fraction to number density [molec cm-3].

	Parameters
	----------
	o3_molmol : float  O3 mole fraction [mol mol-1]
	T_K		  : float  temperature [K]
	P_Pa	  : float  pressure [Pa]

	Returns
	-------
	float  [O3]  [molec cm-3]
	"""
	# Ideal gas:  n/V = P / (kB * T)   with kB = 1.380649e-23 J K-1
	kB = 1.380649e-23					 # J K-1
	n_air_m3 = P_Pa / (kB * T_K)		# molec m-3
	n_air_cm3 = n_air_m3 * 1e-6			# molec cm-3
	return o3_molmol * n_air_cm3


def _calc_pss_ratio_row(lat, lon, timestamp_utc, T_K, d_geoscf):
	"""
	Compute the NOx/NO2 PSS ratio at a single trajectory point.

		NOx/NO2 = 1 + j_NO2 / (k_NO+O3 * [O3])

	GEOS-CF fields accessed
	-----------------------
	chm_tavg_1hr_glo_L1440x721_slv	→  O3  [mol mol-1]	(surface, lev_idx=0)
	xgc_tavg_1hr_glo_L1440x721_slv	→  COSZ [-]
	met_tavg_1hr_glo_L1440x721_slv	→  T [K] and PS [Pa]  (if T_K not supplied)

	Parameters
	----------
	lat			  : float		   [degrees_north]
	lon			  : float		   [degrees_east]
	timestamp_utc : pd.Timestamp   tz-naive UTC (used to find GEOS-CF file)
	T_K			  : float		   temperature [K]	(from tdump["temp"])
								   Pass np.nan to fall back to GEOS-CF T.
	d_geoscf	  : str			   root path for GEOS-CF files (CT.d_geoscf)

	Returns
	-------
	dict with keys:
		pss_ratio	: NOx/NO2  [-]
		jno2		: [s-1]
		k_no_o3		: [cm3 molec-1 s-1]
		o3_molmol	: [mol mol-1]
		o3_nd		: [molec cm-3]
		T_K			: [K]  (source: tdump or GEOS-CF)
		cosz		: [-]
	"""
	ts = pd.Timestamp(timestamp_utc)

	# --- O3 from chm_tavg collection ---
	f_chm = _geoscf_filepath("chm_tavg_1hr_glo_L1440x721_slv", ts, d_geoscf)
	o3_molmol = _geoscf_interp_point(f_chm, "O3", lat, lon, lev_idx=0)

	# --- COSZ from xgc_tavg collection ---
	f_xgc = _geoscf_filepath("xgc_tavg_1hr_glo_L1440x721_slv", ts, d_geoscf)
	cosz  = _geoscf_interp_point(f_xgc, "COSZ", lat, lon)

	# --- Temperature: prefer tdump value, fall back to GEOS-CF met ---
	if np.isnan(T_K):
		f_met = _geoscf_filepath("met_tavg_1hr_glo_L1440x721_slv", ts, d_geoscf)
		T_K   = _geoscf_interp_point(f_met, "T", lat, lon, lev_idx=0)

	# --- Surface pressure for number density ---
	f_met  = _geoscf_filepath("met_tavg_1hr_glo_L1440x721_slv", ts, d_geoscf)
	P_Pa   = _geoscf_interp_point(f_met, "PS", lat, lon)
	if np.isnan(P_Pa):
		P_Pa = _P0_PA	# fallback to standard atmosphere

	# --- Derived quantities ---
	jno2   = _jno2_from_cosz(cosz)
	k	   = _k_no_o3(T_K)
	o3_nd  = _o3_number_density(o3_molmol, T_K, P_Pa)

	denom = k * o3_nd
	pss_ratio = 1.0 + (jno2 / denom) if denom > 0 else np.nan

	return {
		"pss_ratio":  pss_ratio,
		"jno2":		  jno2,
		"k_no_o3":	  k,
		"o3_molmol":  o3_molmol,
		"o3_nd":	  o3_nd,
		"T_K":		  T_K,
		"cosz":		  cosz,
		"P_Pa":		  P_Pa,
	}


def step1_pss_ratio(trop_csf, tdump, d_geoscf, sfx="_O", rsq_min=0.7, d_gap_max=50.0, nobs_min=5):
	"""
	Step 1: Compute the PSS (NOx/NO2) ratio at every trajectory point
	using GEOS-CF j_NO2, O3, and T.

	Adds columns to trop_csf (in-place copy):
		pss_ratio, jno2, k_no_o3, o3_molmol, o3_nd, T_K_geoscf, cosz, P_Pa

	Parameters
	----------
	trop_csf   : pd.DataFrame	output of _calc_csf (with _O or _H suffix)
	tdump	   : pd.DataFrame	true_tdump or original tdump
	d_geoscf   : str			CT.d_geoscf
	sfx			: str			column suffix used in trop_csf ("_O" or "_H")
	rsq_min    : float			minimum rsq_detrend for valid flux rows
	d_gap_max  : float			maximum d_gap [%] for valid flux rows
	nobs_min   : int			minimum nobs for valid flux rows

	Returns
	-------
	trop_csf : pd.DataFrame   with PSS columns appended
	"""
	pss_cols = [i+sfx for i in ["pss_ratio", "jno2", "k_no_o3", "o3_molmol", "o3_nd", "T_K_geoscf", "cosz", "P_Pa"]]
	for c in pss_cols:
		trop_csf[c] = np.nan

	# Merge lat/lon/time_tag from tdump if not already present
	lat_col  = f"lat_a3{sfx}"
	lon_col  = f"lon_a3{sfx}"
	time_col = f"time_tag{sfx}"

	for idx, row in trop_csf.iterrows():
		lat  = row.get(lat_col,  np.nan)
		lon  = row.get(lon_col,  np.nan)
		time = row.get(time_col, pd.NaT)
		T_K  = row.get(f"temp{sfx}", np.nan)   # from tdump["temp"]

		if any(pd.isna(v) for v in [lat, lon, time]):
			continue
		if pd.isna(T_K):
			T_K = np.nan   # will trigger GEOS-CF fallback inside helper

		res = _calc_pss_ratio_row(lat, lon, time, T_K, d_geoscf)
		trop_csf.loc[idx, "pss_ratio"+sfx]	= res["pss_ratio"]
		trop_csf.loc[idx, "jno2"+sfx]		= res["jno2"]
		trop_csf.loc[idx, "k_no_o3"+sfx]	= res["k_no_o3"]
		trop_csf.loc[idx, "o3_molmol"+sfx]	= res["o3_molmol"]
		trop_csf.loc[idx, "o3_nd"+sfx]		= res["o3_nd"]
		trop_csf.loc[idx, "T_K_geoscf"+sfx]	= res["T_K"]
		trop_csf.loc[idx, "cosz"+sfx]		= res["cosz"]
		trop_csf.loc[idx, "P_Pa"+sfx]		= res["P_Pa"]

	return trop_csf


# =============================================================================
# STEP 2 — Identify the PSS point (quality screen + flux peak)
# =============================================================================
def step2_find_pss_point(trop_csf, sfx="_O", residual_tol=2.0, rsq_min=0.7, d_gap_max=50.0, nobs_min=5):	 # fallback only
	"""
	Step 2: Screen flux values by quality flags and locate the trajectory
	point where flux_no2 peaks	→  this is the PSS point.

	Quality criteria (editable via arguments):
		rsq_detrend{suffix}  >=  rsq_min		(Gaussian fit quality)
		d_gap{suffix}		 <=  d_gap_max [%]	 (sampling gap fraction)
		nobs{suffix}		 >=  nobs_min		  (number of pixels)

	Parameters
	----------
	trop_csf			: pd.DataFrame	output of step1_pss_ratio
	suffix				: str			"_O" or "_H"
	rsq_min				: float
	d_gap_max			: float
	nobs_min			: int

	Returns
	-------
	trop_csf_qc  : pd.DataFrame   quality-screened subset
	pss_row		 : pd.Series	  row where flux_no2 is maximum (PSS point)
	pss_tdump_id : int			  tdump_id of the PSS point
	"""

	## 2a. Quality filtering
	mask = trop_csf['flux_no2'+sfx].notna() & (trop_csf['flux_no2'+sfx] > 0)
	mask &= trop_csf['QF_gauss_abs'+sfx] == 0
	trop_csf_qc = trop_csf.loc[mask].sort_values("tdump_id").reset_index(drop=True)

	if trop_csf_qc.empty or 'flux_no2'+sfx not in trop_csf_qc.columns:
		print(f"  [step2] No valid flux rows after QC (sfx={sfx})")
		return trop_csf_qc, pd.Series(dtype=float), np.nan

	## 2b. Identify the primary NO2 plume by fitting a log-normal pulse to (age,flux) and interatively rejecting outliers with large residuals
	def _fit_primary_plume(flux, age, residual_tol=2.0):
		"""
		This is robust to:
		  - gaps in tdump_id (fit is global, not point-by-point)
		  - spurious large flux from a separate downwind source (outlier rejection)
		  - noise on individual transects (least-squares fit averages over all)
		Parameters
		----------
		flux		 : np.ndarray  shape (N,) QC-screened flux, ascending age order
		age			 : np.ndarray  shape (N,) transport age [hours], same order
		residual_tol : float	   points with |residual| > residual_tol * MAD of residuals are flagged as outside primary plume; lower = stricter outlier rejection (default 2.0)
		Returns
		-------
		in_primary	 : np.ndarray  bool mask, shape (N,)  True = primary plume
		t_peak_fit	 : float	   fitted peak age [hours]	(PSS point estimate)
		fit_ok		 : bool		   False if fitting failed (caller falls back to argmax)
		"""
		# Log-normal pulse function
		def _lognormal_pulse(t, A, t_peak, sigma):
			"""
			Unimodal log-normal pulse shape for NO2 flux vs. transport age. f(t) = A * exp( -(ln(t / t_peak))^2 / (2 * sigma^2) ). Defined only for t > 0; returns 0 elsewhere.
			"""
			t	  = np.asarray(t, dtype=float)
			out   = np.zeros_like(t)
			valid = t > 0
			out[valid] = A * np.exp(-(np.log(t[valid] / t_peak))**2 / (2 * sigma**2))
			return out

		# Need at least 4 points to fit 3 parameters with a degree of freedom
		if len(flux) < 4:
			return np.ones(n, dtype=bool), age[np.argmax(flux)], False

		# Initial parameter guess
		i0		= np.argmax(flux)
		A0		= flux[i0]
		t_peak0 = max(age[i0], 0.1)
		sigma0	= 0.5	# log-normal width; 0.5 covers ~1 decade in age

		# Fit a log-normal pulse to all points
		try:
			popt, _ = curve_fit( _lognormal_pulse, age, flux, p0=[A0, t_peak0, sigma0], bounds=([0, 0.01, 0.05], [A0 * 5, age.max() * 1.5, 3.0]), maxfev=5000, )
			A_fit, t_peak_fit, sigma_fit = popt
		except RuntimeError:
			return np.ones(n, dtype=bool), age[np.argmax(flux)], False

		# Iterative outlier rejection: flag points whose residual exceeds residual_tol × MAD (median absolute deviation) of all residuals.
		residuals	= flux - _lognormal_pulse(age, *popt)
		mad			= np.median(np.abs(residuals - np.median(residuals)))
		in_primary	= np.abs(residuals) <= residual_tol * (mad + 1e-9)

		# Refit on inliers only if any points were rejected
		if not np.all(in_primary) and in_primary.sum() >= 4:
			try:
				popt2, _ = curve_fit( _lognormal_pulse, age[in_primary], flux[in_primary], p0=popt, bounds=([0, 0.01, 0.05], [A_fit * 5, age.max() * 1.5, 3.0]), maxfev=5000, )
				A_fit, t_peak_fit, sigma_fit = popt2
				residuals  = flux - _lognormal_pulse(age, *popt2)
				mad		   = np.median(np.abs(residuals - np.median(residuals)))
				in_primary = np.abs(residuals) <= residual_tol * (mad + 1e-9)
			except RuntimeError:
				pass   # keep first-pass result

		return in_primary, float(t_peak_fit), True

	in_primary, t_peak_fit, fit_ok	= _fit_primary_plume(flux= trop_csf_qc['flux_no2'+sfx].values, age= np.abs(trop_csf_qc['age_hours'+sfx].values), residual_tol=2.0)

	trop_csf_qc["step2_in_primary_plume"+sfx] = in_primary
	n_excl = (~in_primary).sum()
	if n_excl > 0:
		print(f"  [step2] Outlier rejection: excluded {n_excl}/{len(trop_csf_qc)} rows outside primary plume fit  (fit_ok={fit_ok})")

	primary = trop_csf_qc.loc[in_primary]

	# If outlier rejection removed everything, fall back to full QC set
	if primary.empty:
		print(f"  [step2] All rows excluded by outlier rejection — falling back to full QC set.")
		primary = trop_csf_qc
		fit_ok	= False   # force argmax fallback

	## 2c. Find PSS point: data row closest to fitted t_peak (or argmax if fit failed)
	if fit_ok:
		age_abs		= np.abs(primary['age_hours'+sfx].values)
		closest_idx = primary.index[np.argmin(np.abs(age_abs - t_peak_fit))]
		pss_row		= trop_csf_qc.loc[closest_idx]
	else:
		pss_row = primary.loc[primary['flux_no2'+sfx].idxmax()]

	pss_tdump_id = pss_row.get("tdump_id", np.nan)
	print(f"  [step2] PSS point: tdump_id={pss_tdump_id}  "
		  f"flux_no2={pss_row['flux_no2'+sfx]:.4f} tNO2/hr	"
		  f"age_hrs={pss_row.get('age_hours'+sfx, np.nan):.2f}	"
		  f"t_peak_fit={t_peak_fit:.2f} hr	fit_ok={fit_ok}")

	return trop_csf_qc, pss_row, pss_tdump_id

# =============================================================================
# STEP 3 — Convert NO2 flux → NOx flux at the PSS point
# =============================================================================
def step3_no2_to_nox_flux(pss_row, sfx):
	"""
	Step 3: Multiply the NO2 flux at the PSS point by the PSS (NOx/NO2) ratio to obtain the NOx flux at the PSS point.

		flux_nox_PSS = flux_no2_PSS  ×	(NOx/NO2)_PSS

	Parameters
	----------
	pss_row				: pd.Series   row returned by step2_find_pss_point
	suffix				: str
	flux_col_template	: str

	Returns
	-------
	flux_nox_PSS : float   [tNOx/hr]  (same units as flux_no2 in _calc_csf)
	pss_ratio	 : float   NOx/NO2 at PSS
	"""
	flux_no2   = float(pss_row.get('flux_no2'+sfx, np.nan))
	pss_ratio  = float(pss_row.get("pss_ratio"+sfx, np.nan))

	if np.isnan(flux_no2) or np.isnan(pss_ratio):
		print(f"  [step3] Cannot compute NOx flux: " f"flux_no2={flux_no2:.4f}	pss_ratio={pss_ratio:.4f}")
		return np.nan, np.nan

	# Unit note: flux_no2 is in tNO2/hr (from _calc_csf).
	# pss_ratio is dimensionless (NOx/NO2 by mole).
	# Because MW(NOx) ≈ MW(NO2) in the PSS approximation (NO2 dominates),
	# multiplying by the molar ratio gives tNOx/hr to good approximation.
	# For a strict mass conversion: multiply additionally by MW_NO/MW_NO2
	# for the NO fraction; here we follow the standard CSF convention.
	flux_nox_PSS = flux_no2 * pss_ratio

	print(f"  [step3] flux_no2_PSS = {flux_no2:.4f} tNO2/hr  × pss_ratio = {pss_ratio:.3f}  → flux_nox_PSS = {flux_nox_PSS:.4f} tNOx/hr")

	return flux_nox_PSS, pss_ratio


# =============================================================================
# STEP 4 — Decay correction: back-extrapolate NOx flux to t = 0 (source)
# =============================================================================

def _get_oh_column(lat, lon, timestamp_utc, d_geoscf):
	"""
	Return a column-averaged [OH] [molec cm-3] at (lat, lon, time)
	from the GEOS-CF oxi_inst_1hr_glo_L1440x721_v72 collection.

	Uses the surface (bottom) model level as a boundary-layer representative
	value for the OH reactivity calculation.  For a mass-weighted column
	average, integrate over DELP from met_inst_1hr_glo_L1440x721_v72.
	"""
	ts	   = pd.Timestamp(timestamp_utc)
	f_oxi  = _geoscf_filepath("oxi_inst_1hr_glo_L1440x721_v72", ts, d_geoscf, mode="ana")
	f_met  = _geoscf_filepath("met_inst_1hr_glo_L1440x721_v72", ts, d_geoscf, mode="ana")

	oh_col	 = _geoscf_interp_column(f_oxi, "OH",	lat, lon)	# (72,) mol mol-1
	delp_col = _geoscf_interp_column(f_met, "DELP", lat, lon)	# (72,) Pa
	T_col	 = _geoscf_interp_column(f_met, "T",	lat, lon)	# (72,) K
	P_col	 = _geoscf_interp_column(f_met, "PL",	lat, lon)	# (72,) Pa

	if oh_col is None or delp_col is None:
		# Fall back to surface value only
		oh_sfc = _geoscf_interp_point(f_oxi, "OH", lat, lon, lev_idx=-1)
		T_sfc  = _geoscf_interp_point(f_met, "T",  lat, lon, lev_idx=-1)
		P_sfc  = _geoscf_interp_point(f_met, "PS", lat, lon)
		kB	   = 1.380649e-23
		n_air  = P_sfc / (kB * T_sfc) * 1e-6	 # molec cm-3
		return oh_sfc * n_air

	# Mass-weighted column average OH number density
	kB	  = 1.380649e-23
	n_air = P_col / (kB * T_col) * 1e-6			  # molec cm-3	(72,)
	oh_nd = oh_col * n_air						   # molec cm-3  (72,)
	# weights = pressure thickness
	oh_avg = np.average(oh_nd, weights=np.abs(delp_col))
	return float(oh_avg)


def _k_loss_nox(T_K, oh_nd_cm3):
	"""
	Effective first-order NOx loss rate [s-1] via
		NO2 + OH + M → HNO3
	using the JPL 2019 termolecular rate constant evaluated at the
	boundary-layer pressure (~1000 hPa, M ~ 2.5e19 molec cm-3).

	k_eff = k_NO2+OH × [OH]

	k_NO2+OH (termolecular, high-pressure limit at 1000 hPa):
		k_0 = 2.4e-30 × (T/300)^-3.0   [cm6 molec-2 s-1]
		k_inf = 1.7e-11 × (T/300)^-2.1 [cm3 molec-1 s-1]
		Fc = 0.6
		k = k_0*[M] / (1 + k_0*[M]/k_inf) × Fc^{1/(1+(log(k_0*[M]/k_inf))^2)}
	(JPL 19-5, Table 2-1)

	Parameters
	----------
	T_K		  : float  [K]
	oh_nd_cm3 : float  [OH] [molec cm-3]

	Returns
	-------
	k_loss : float	[s-1]
	"""
	M	 = 2.5e19	# air number density [molec cm-3] at ~1000 hPa, 298 K
	Fc	 = 0.6
	Tr	 = T_K / 300.0
	k0	 = 2.4e-30 * Tr ** (-3.0) * M		 # [cm3 molec-1 s-1] × [M]
	kinf = 1.7e-11 * Tr ** (-2.1)
	k_f  = k0 / (1.0 + k0 / kinf)
	N	 = 1.0 + (np.log10(k0 / kinf)) ** 2
	k_NO2_OH = k_f * Fc ** (1.0 / N)		  # [cm3 molec-1 s-1]
	return k_NO2_OH * oh_nd_cm3				   # [s-1]


def step4_decay_correction(trop_csf_qc, pss_row, flux_nox_PSS, d_geoscf, suffix="_O", flux_col_template="flux_no2{s}", option="B"):
	"""
	Step 4: Back-extrapolate NOx flux from the PSS point to t=0 (source).

	Option A — Empirical (fit exponential to post-PSS flux decline):
		Fits  flux_nox(t) = flux_nox_PSS × exp(-k_eff × t)
		to QC-screened post-PSS rows, then evaluates at age_PSS.

	Option B — Analytical (k_loss from GEOS-CF [OH]):
		k_loss from [OH] climatology (oxi_inst collection),
		then:  flux_nox_source = flux_nox_PSS × exp(+k_loss × age_at_PSS)

	Both options return flux_nox_source [tNOx/hr].

	Parameters
	----------
	trop_csf_qc			: pd.DataFrame	QC-screened CSF table (from step2)
	pss_row				: pd.Series		PSS point row (from step2)
	flux_nox_PSS		: float			[tNOx/hr]  (from step3)
	d_geoscf			: str			CT.d_geoscf
	suffix				: str			"_O" or "_H"
	flux_col_template	: str
	option				: str			"A" or "B"

	Returns
	-------
	dict with keys:
		flux_nox_source   : float  [tNOx/hr]
		k_loss			  : float  [s-1]   (nan for option A)
		k_eff			  : float  [s-1]   (nan for option B)
		age_at_PSS_s	  : float  [s]
		option			  : str    "A" or "B"
		oh_nd			  : float  [molec cm-3]  (nan for option A)
	"""
	age_col = f"age_hours{suffix}"
	flux_col = flux_col_template.format(s=suffix)

	age_pss_hr = float(pss_row.get(age_col, np.nan))
	age_pss_s  = abs(age_pss_hr) * 3600.0	   # HYSPLIT back-runs: negative → abs

	result = {
		"flux_nox_source": np.nan,
		"k_loss":		   np.nan,
		"k_eff":		   np.nan,
		"age_at_PSS_s":    age_pss_s,
		"option":		   option,
		"oh_nd":		   np.nan,
		"flux_nox_fit":    None,   # Series (same index as trop_csf_qc) set below
	}

	if np.isnan(flux_nox_PSS) or np.isnan(age_pss_s):
		print("  [step4] Missing flux_nox_PSS or age_at_PSS — skipping.")
		return result

	# Helper: evaluate fitted curve at every QC row age and store as Series
	def _eval_fit_curve(k, t_ref_s=age_pss_s):
		"""flux_nox(t) = flux_nox_PSS * exp(-k * (t - t_ref))  evaluated per row."""
		t_s = np.abs(trop_csf_qc[age_col].values) * 3600.0
		return pd.Series(flux_nox_PSS * np.exp(-k * (t_s - t_ref_s)), index=trop_csf_qc.index)

	# ------------------------------------------------------------------ #
	# Option A: empirical exponential fit to post-PSS flux decline		  #
	# ------------------------------------------------------------------ #
	if option == "A":
		pss_tid = pss_row.get("tdump_id", -1)
		post = trop_csf_qc.loc[trop_csf_qc["tdump_id"] > pss_tid].copy()

		if flux_col not in post.columns:
			print("  [step4-A] No post-PSS flux data for fit.")
			return result

		post["flux_nox"] = post[flux_col] * post.get("pss_ratio", 1.0)
		post = post.dropna(subset=[age_col, "flux_nox"]).loc[lambda d: d["flux_nox"] > 0]

		if len(post) < 3:
			print(f"  [step4-A] Insufficient post-PSS points (n={len(post)}).")
			return result

		# Fit  f_rel(t_rel) = exp(-k * t_rel)  where t_rel = t - age_pss_s
		t_rel = np.abs(post[age_col].values) * 3600.0 - age_pss_s
		f_rel = post["flux_nox"].values / flux_nox_PSS
		try:
			popt, _ = curve_fit(lambda t, k: np.exp(-k * t), t_rel, f_rel,
								p0=[1e-4], bounds=([0], [1e-1]), maxfev=10_000)
			k_eff = float(popt[0])
		except RuntimeError:
			print("  [step4-A] Exponential fit failed.")
			return result

		flux_nox_source = flux_nox_PSS * np.exp(k_eff * age_pss_s)
		result.update({"flux_nox_source": flux_nox_source,
					   "k_eff":			  k_eff,
					   "flux_nox_fit":	  _eval_fit_curve(k_eff)})
		print(f"  [step4-A] k_eff={k_eff:.2e} s-1  age_PSS={age_pss_s/3600:.2f} hr  flux_nox_source={flux_nox_source:.4f} tNOx/hr")

	# ------------------------------------------------------------------ #
	# Option B: analytical — k_loss from GEOS-CF [OH]					 #
	# ------------------------------------------------------------------ #
	elif option == "B":
		lat  = float(pss_row.get(f"lat_a3{suffix}", np.nan))
		lon  = float(pss_row.get(f"lon_a3{suffix}", np.nan))
		time = pss_row.get(f"time_tag{suffix}", pd.NaT)
		T_K  = float(pss_row.get("T_K_geoscf", np.nan))

		if any(pd.isna(v) for v in [lat, lon]):
			print("  [step4-B] Missing lat/lon for OH look-up.")
			return result

		oh_nd = _get_oh_column(lat, lon, time, d_geoscf)
		if np.isnan(T_K):
			T_K = float(pss_row.get(f"temp{suffix}", 298.0))

		k_loss			= _k_loss_nox(T_K, oh_nd)
		flux_nox_source = flux_nox_PSS * np.exp(k_loss * age_pss_s)
		result.update({"flux_nox_source": flux_nox_source,
					   "k_loss":		  k_loss,
					   "oh_nd":			  oh_nd,
					   "flux_nox_fit":	  _eval_fit_curve(k_loss)})
		print(f"  [step4-B] [OH]={oh_nd:.2e} molec/cm3	k_loss={k_loss:.2e} s-1  age_PSS={age_pss_s/3600:.2f} hr  flux_nox_source={flux_nox_source:.4f} tNOx/hr")

	return result


# =============================================================================
# MAIN CALLER — drop into _csf_prcs() after step 2i (CSV saved)
# =============================================================================
def run_nox_workflow(trop_csf, true_tdump, d_geoscf, suffix="_O", option_b_or_a="B", residual_tol=2.0):
	"""
	Orchestrates Steps 1–5.  Call this inside _csf_prcs() after trop_csf.to_csv(...)

		trop_csf, nox_result = run_nox_workflow(
			trop_csf, true_tdump, CT.d_geoscf, suffix="_O", option_b_or_a="B"
		)

	trop_csf gains per-row columns from Step 1 (pss_ratio, jno2, k_no_o3,
	o3_molmol, o3_nd, T_K_geoscf, cosz, P_Pa) and a QC/PSS flag column from
	Step 2 (step2_qc_pass, step2_is_pss_point).  All scalar step results
	(Steps 2–5) are returned in nox_result only — not broadcast as columns.

	Returns
	-------
	trop_csf   : pd.DataFrame  with Step 1–2 per-row columns appended
	nox_result : dict		   scalars from all five steps; keys:
					 status, pss_tdump_id, pss_ratio, flux_nox_PSS,
					 flux_nox_source, k_loss, k_eff, age_at_PSS_s, oh_nd, option
	"""
	print("  [NOx workflow] Starting 5-step PSS back-calculation ...")

	# Step 1 — per-row PSS fields (adds columns directly to trop_csf)
	trop_csf = step1_pss_ratio(trop_csf, true_tdump, d_geoscf, sfx=suffix)

	# Step 2 — QC screen + locate PSS point
	trop_csf_qc, pss_row, pss_tdump_id = step2_find_pss_point(trop_csf, suffix=suffix)
	trop_csf["step2_qc_pass"+suffix]	  = trop_csf.index.isin(trop_csf_qc.index)
	trop_csf["step2_is_pss_point"+suffix] = trop_csf["tdump_id"] == pss_tdump_id

	if pss_row.empty or np.isnan(pss_tdump_id):
		print("  [NOx workflow] No valid PSS point found — aborting.")
		return trop_csf, {"status": "no_pss_point"}

	# Step 3 — NO2 flux → NOx flux at PSS (scalars only)
	flux_nox_PSS, pss_ratio = step3_no2_to_nox_flux(pss_row, suffix=suffix)
	if np.isnan(flux_nox_PSS):
		print("  [NOx workflow] Step 3 failed — aborting.")
		return trop_csf, {"status": "step3_failed"}

	# Step 4 — decay correction back to t=0
	decay = step4_decay_correction( trop_csf_qc, pss_row, flux_nox_PSS, d_geoscf, suffix=suffix, option=option_b_or_a)

	# flux_nox_fit is per-row (QC rows only) → write into trop_csf column
	trop_csf["flux_nox_fit"+suffix] = np.nan
	if decay["flux_nox_fit"] is not None:
		trop_csf.loc[decay["flux_nox_fit"].index, "flux_nox_fit"+suffix] = decay["flux_nox_fit"].values

	# Step 5 — aggregate scalar nox_result (flux_nox_fit excluded; it lives in trop_csf)
	nox_result = {
		"status"+suffix:		   "ok",
		"pss_tdump_id"+suffix:    pss_tdump_id,
		"pss_ratio"+suffix:	   pss_ratio,
		"flux_nox_PSS"+suffix:    flux_nox_PSS,
		**{k: v for k, v in decay.items() if k != "flux_nox_fit"},
	}
	print("  [NOx workflow] Complete.")
	return trop_csf, nox_result

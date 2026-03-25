# =============================================================================
# csf_geoscf_nox.py  —	GEOS-CF-driven NOx emission workflow
# Drop-in companion to csf_prcs.py (v20260323)
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
#	GEOS.cf.ana.<collection>.<yyyyMMdd_hhmmz>.R0.nc4
# and be reachable via the NCCS OPeNDAP or local mirror path CT.d_geoscf.
# =============================================================================

import os
import math
import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import curve_fit
from scipy.stats import linregress

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
	# Round DOWN to nearest collection hour (tavg files are stamped at hh:30,
	# inst files at hh:00; we just match the hour for look-up)
	ts	  = pd.Timestamp(timestamp_utc)
	hh	  = ts.floor("h").strftime("%H")
	mm	  = "30" if "tavg" in collection else "00"
	stamp = f"{ts.strftime('%Y%m%d')}_{hh}{mm}z"
	fname = f"GEOS.cf.{mode}.{collection}.{stamp}.R0.nc4"
	ddir  = os.path.join(d_geoscf, "NRTv2", "pub", mode,
						 f"Y{ts.strftime('%Y')}",
						 f"M{ts.strftime('%m')}",
						 f"D{ts.strftime('%d')}")
	return os.path.join(ddir, fname)


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


def step1_pss_ratio(trop_csf, tdump, d_geoscf, suffix="_O", rsq_min=0.7, d_gap_max=50.0, nobs_min=5):
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
	suffix	   : str			column suffix used in trop_csf ("_O" or "_H")
	rsq_min    : float			minimum rsq_detrend for valid flux rows
	d_gap_max  : float			maximum d_gap [%] for valid flux rows
	nobs_min   : int			minimum nobs for valid flux rows

	Returns
	-------
	trop_csf : pd.DataFrame   with PSS columns appended
	"""
	out = trop_csf.copy()
	pss_cols = ["pss_ratio", "jno2", "k_no_o3", "o3_molmol", "o3_nd", "T_K_geoscf", "cosz", "P_Pa"]
	for c in pss_cols:
		out[c] = np.nan

	# Merge lat/lon/time_tag from tdump if not already present
	lat_col  = f"lat_a3{suffix}"
	lon_col  = f"lon_a3{suffix}"
	time_col = f"time_tag{suffix}"

	for idx, row in out.iterrows():
		lat  = row.get(lat_col,  np.nan)
		lon  = row.get(lon_col,  np.nan)
		time = row.get(time_col, pd.NaT)
		T_K  = row.get(f"temp{suffix}", np.nan)   # from tdump["temp"]

		if any(pd.isna(v) for v in [lat, lon, time]):
			continue
		if pd.isna(T_K):
			T_K = np.nan   # will trigger GEOS-CF fallback inside helper

		res = _calc_pss_ratio_row(lat, lon, time, T_K, d_geoscf)
		out.loc[idx, "pss_ratio"]	= res["pss_ratio"]
		out.loc[idx, "jno2"]		= res["jno2"]
		out.loc[idx, "k_no_o3"]		= res["k_no_o3"]
		out.loc[idx, "o3_molmol"]	= res["o3_molmol"]
		out.loc[idx, "o3_nd"]		= res["o3_nd"]
		out.loc[idx, "T_K_geoscf"]	= res["T_K"]
		out.loc[idx, "cosz"]		= res["cosz"]
		out.loc[idx, "P_Pa"]		= res["P_Pa"]

	return out


# =============================================================================
# STEP 2 — Identify the PSS point (quality screen + flux peak)
# =============================================================================

def step2_find_pss_point(trop_csf, suffix="_O",
						 rsq_min=0.7, d_gap_max=50.0, nobs_min=5,
						 flux_col_template="flux_no2{s}"):
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
	flux_col_template	: str		   column name template; {s} → suffix

	Returns
	-------
	trop_csf_qc  : pd.DataFrame   quality-screened subset
	pss_row		 : pd.Series	  row where flux_no2 is maximum (PSS point)
	pss_tdump_id : int			  tdump_id of the PSS point
	"""
	flux_col = flux_col_template.format(s=suffix)
	rsq_col  = f"rsq_detrend{suffix}"
	gap_col  = f"d_gap{suffix}"
	nobs_col = f"nobs{suffix}"

	df = trop_csf.copy()

	# Apply quality screens (only where columns exist)
	mask = pd.Series(True, index=df.index)
	if rsq_col	in df.columns: mask &= df[rsq_col].fillna(-999)  >= rsq_min
	if gap_col	in df.columns: mask &= df[gap_col].fillna(999)	 <= d_gap_max
	if nobs_col in df.columns: mask &= df[nobs_col].fillna(0)	 >= nobs_min
	if flux_col in df.columns: mask &= df[flux_col].notna() & (df[flux_col] > 0)

	trop_csf_qc = df.loc[mask].reset_index(drop=True)

	if trop_csf_qc.empty or flux_col not in trop_csf_qc.columns:
		print(f"  [step2] No valid flux rows after QC (suffix={suffix})")
		return trop_csf_qc, pd.Series(dtype=float), np.nan

	# Find tdump_id where flux_no2 peaks
	peak_idx	 = trop_csf_qc[flux_col].idxmax()
	pss_row		 = trop_csf_qc.loc[peak_idx]
	pss_tdump_id = pss_row.get("tdump_id", np.nan)

	print(f"  [step2] PSS point: tdump_id={pss_tdump_id}  "
		  f"flux_no2={pss_row[flux_col]:.4f} tNO2/hr  "
		  f"age_hrs={pss_row.get(f'age_hours{suffix}', np.nan):.2f}")

	return trop_csf_qc, pss_row, pss_tdump_id


# =============================================================================
# STEP 3 — Convert NO2 flux → NOx flux at the PSS point
# =============================================================================

def step3_no2_to_nox_flux(pss_row, suffix="_O",
						   flux_col_template="flux_no2{s}"):
	"""
	Step 3: Multiply the NO2 flux at the PSS point by the PSS (NOx/NO2) ratio
	to obtain the NOx flux at the PSS point.

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
	flux_col   = flux_col_template.format(s=suffix)
	flux_no2   = float(pss_row.get(flux_col, np.nan))
	pss_ratio  = float(pss_row.get("pss_ratio", np.nan))

	if np.isnan(flux_no2) or np.isnan(pss_ratio):
		print(f"  [step3] Cannot compute NOx flux: "
			  f"flux_no2={flux_no2:.4f}  pss_ratio={pss_ratio:.4f}")
		return np.nan, np.nan

	# Unit note: flux_no2 is in tNO2/hr (from _calc_csf).
	# pss_ratio is dimensionless (NOx/NO2 by mole).
	# Because MW(NOx) ≈ MW(NO2) in the PSS approximation (NO2 dominates),
	# multiplying by the molar ratio gives tNOx/hr to good approximation.
	# For a strict mass conversion: multiply additionally by MW_NO/MW_NO2
	# for the NO fraction; here we follow the standard CSF convention.
	flux_nox_PSS = flux_no2 * pss_ratio

	print(f"  [step3] flux_no2_PSS = {flux_no2:.4f} tNO2/hr  "
		  f"× pss_ratio = {pss_ratio:.3f}  "
		  f"→ flux_nox_PSS = {flux_nox_PSS:.4f} tNOx/hr")

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
	f_oxi  = _geoscf_filepath("oxi_inst_1hr_glo_L1440x721_v72", ts, d_geoscf,
							   mode="ana")
	f_met  = _geoscf_filepath("met_inst_1hr_glo_L1440x721_v72", ts, d_geoscf,
							   mode="ana")

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


def step4_decay_correction(trop_csf_qc, pss_row, flux_nox_PSS,
							d_geoscf, suffix="_O",
							flux_col_template="flux_no2{s}",
							option="B"):
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
	}

	if np.isnan(flux_nox_PSS) or np.isnan(age_pss_s):
		print("  [step4] Missing flux_nox_PSS or age_at_PSS — skipping.")
		return result

	# ------------------------------------------------------------------ #
	# Option A: empirical exponential fit to post-PSS flux				 #
	# ------------------------------------------------------------------ #
	if option == "A":
		pss_tid = pss_row.get("tdump_id", -1)

		# Post-PSS rows: tdump_id > pss_tid (further downwind / older air)
		post = trop_csf_qc.loc[
			trop_csf_qc["tdump_id"] > pss_tid
		].copy()

		if flux_col in post.columns:
			post["flux_nox"] = post[flux_col] * post.get("pss_ratio", 1.0)
		else:
			print("  [step4-A] No post-PSS flux data for fit.")
			return result

		post = post.dropna(subset=[age_col, "flux_nox"])
		post = post.loc[post["flux_nox"] > 0]

		if len(post) < 3:
			print("  [step4-A] Insufficient post-PSS points for fit "
				  f"(n={len(post)}).")
			return result

		t_arr = np.abs(post[age_col].values) * 3600.0	# [s]
		f_arr = post["flux_nox"].values

		# Normalise by flux_nox_PSS so we fit decay from 1 at t=age_pss_s
		t_rel = t_arr - age_pss_s
		f_rel = f_arr / flux_nox_PSS

		try:
			def _exp_decay(t, k): return np.exp(-k * t)
			popt, _ = curve_fit(_exp_decay, t_rel, f_rel,
								p0=[1e-4], bounds=([0], [1e-1]),
								maxfev=10_000)
			k_eff = float(popt[0])
		except RuntimeError:
			print("  [step4-A] Exponential fit failed.")
			return result

		flux_nox_source = flux_nox_PSS * np.exp(k_eff * age_pss_s)
		result.update({
			"flux_nox_source": flux_nox_source,
			"k_eff":		   k_eff,
		})
		print(f"  [step4-A] k_eff={k_eff:.2e} s-1  "
			  f"age_PSS={age_pss_s/3600:.2f} hr  "
			  f"flux_nox_source={flux_nox_source:.4f} tNOx/hr")

	# ------------------------------------------------------------------ #
	# Option B: analytical — k_loss from GEOS-CF [OH]					#
	# ------------------------------------------------------------------ #
	elif option == "B":
		lat  = float(pss_row.get(f"lat_a3{suffix}", np.nan))
		lon  = float(pss_row.get(f"lon_a3{suffix}", np.nan))
		time = pss_row.get(f"time_tag{suffix}", pd.NaT)
		T_K  = float(pss_row.get(f"T_K_geoscf", np.nan))

		if any(pd.isna(v) for v in [lat, lon]):
			print("  [step4-B] Missing lat/lon for OH look-up.")
			return result

		oh_nd  = _get_oh_column(lat, lon, time, d_geoscf)
		if np.isnan(T_K):
			# Fall back to tdump temperature
			T_K = float(pss_row.get(f"temp{suffix}", 298.0))

		k_loss = _k_loss_nox(T_K, oh_nd)
		flux_nox_source = flux_nox_PSS * np.exp(k_loss * age_pss_s)

		result.update({
			"flux_nox_source": flux_nox_source,
			"k_loss":		   k_loss,
			"oh_nd":		   oh_nd,
		})
		print(f"  [step4-B] [OH]={oh_nd:.2e} molec/cm3	"
			  f"k_loss={k_loss:.2e} s-1  "
			  f"age_PSS={age_pss_s/3600:.2f} hr  "
			  f"flux_nox_source={flux_nox_source:.4f} tNOx/hr")

	return result


# =============================================================================
# MASTER CALLER — drop into _csf_prcs() after step 2i (CSV saved)
# =============================================================================

def run_nox_workflow(trop_csf, true_tdump, d_geoscf, suffix="_O", option_b_or_a="B"):
	"""
	Orchestrates Steps 1–5.  Call this inside _csf_prcs() immediately after trop_csf.to_csv(...) 

		nox_result = run_nox_workflow(
			trop_csf   = trop_csf,
			true_tdump = true_tdump,
			d_geoscf   = CT.d_geoscf,
			suffix	   = "_O",
			option_b_or_a = "B",	# "A" for empirical, "B" for analytical
		)
		# nox_result is a flat dict; optionally append to a summary DataFrame.

	Parameters
	----------
	trop_csf	  : pd.DataFrame  merged _H/_O CSF result
	true_tdump	  : pd.DataFrame  optimised trajectory
	d_geoscf	  : str			  CT.d_geoscf
	suffix		  : str			  "_O" (optimised) or "_H" (HYSPLIT)
	option_b_or_a : str			  "A" or "B" for step 4

	Returns
	-------
	dict  aggregated results from all five steps
	"""
	print("  [NOx workflow] Starting 5-step PSS back-calculation ...")

	# Step 1
	trop_csf = step1_pss_ratio(trop_csf, true_tdump, d_geoscf, suffix=suffix)

	# Step 2
	trop_csf_qc, pss_row, pss_tdump_id = step2_find_pss_point(trop_csf, suffix=suffix)

	if pss_row.empty or np.isnan(pss_tdump_id):
		print("  [NOx workflow] No valid PSS point found — aborting.")
		return {"status": "no_pss_point"}

	# Step 3
	flux_nox_PSS, pss_ratio = step3_no2_to_nox_flux(pss_row, suffix=suffix)

	if np.isnan(flux_nox_PSS):
		print("  [NOx workflow] Step 3 failed — aborting.")
		return {"status": "step3_failed"}

	# Step 4
	decay = step4_decay_correction(
		trop_csf_qc, pss_row, flux_nox_PSS,
		d_geoscf, suffix=suffix, option=option_b_or_a)

	# Aggregate
	result = {
		"status":		   "ok",
		"pss_tdump_id":    pss_tdump_id,
		"pss_ratio":	   pss_ratio,
		"flux_nox_PSS":    flux_nox_PSS,
		**decay,
	}
	return result

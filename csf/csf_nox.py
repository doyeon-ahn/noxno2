# =============================================================================
# csf_nox.py  —  GEOS-CF-driven NOx emission workflow
# =============================================================================
# Implements a five-step PSS back-calculation to estimate NOx source flux
# from TROPOMI NO2 cross-sectional flux measurements.
#
# GEOS-CF collections used
# -------------------------
#   chm_tavg_1hr_glo_L1440x721_slv  →  O3   [mol mol-1]
#   met_tavg_1hr_glo_L1440x721_slv  →  T    [K],  PS  [Pa]
#   xgc_tavg_1hr_glo_L1440x721_slv  →  COSZ [-]   (used to derive J_NO2)
#   oxi_inst_1hr_glo_L1440x721_v72  →  OH   [mol mol-1]  (Step 4 Option B)
#
# File naming convention (GEOS-CF v2 NRTv2 and hindcast archive)
#   NRTv2 (Aug 2025–):      GEOS.cf.ana.<collection>.<yyyyMMdd_hhmmz>*.nc4
#   Hindcast (2023–Oct 2024): CF2_hindcast.<collection>.<yyyyMMdd_hhmmz>*.nc4
#
# Public entry point:  run_nox_workflow(trop_csf, true_tdump, d_geoscf, ...)
# =============================================================================

import os, glob
import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import curve_fit


# =============================================================================
# PHYSICAL / CHEMICAL CONSTANTS
# =============================================================================

_kB          = 1.380649e-23   # Boltzmann constant          [J K-1]
_P0_PA       = 101325.0       # standard sea-level pressure  [Pa]
_JNO2_OVER   = 8.5e-3         # J_NO2 at overhead sun, clear-sky BL  [s-1]
                               # (Sander et al. 2011)
_A_NO_O3     = 3.0e-12        # NO+O3 rate pre-exponential   [cm3 molec-1 s-1]
_Ea_NO_O3    = 1500.0         # NO+O3 activation temperature [K]
_MW_NO2      = 46.0055        # molecular weight NO2         [g mol-1]
_MW_NO       = 30.0061        # molecular weight NO          [g mol-1]
_M_BL        = 2.5e19         # air number density ~1000 hPa [molec cm-3]


# =============================================================================
# GEOS-CF FILE HELPERS
# =============================================================================

def _geoscf_path(collection, ts, d_geoscf, mode="ana"):
    """
    Return the full path to a GEOS-CF v2 NetCDF-4 file, or "" if not found.

    Handles two archive periods:
      Hindcast  Jan 2023 – Oct 2024  →  CF2_hindcast.<collection>.<stamp>*.nc4
      NRTv2     Aug 2025 – present   →  GEOS.cf.<mode>.<collection>.<stamp>*.nc4
      Gap period (Nov 2024 – Jul 2025) has no data; returns "".

    tavg files are stamped at hh:30z; inst files at hh:00z.
    """
    ts   = pd.Timestamp(ts)
    hh   = ts.floor("h").strftime("%H")
    mm   = "30" if "tavg" in collection else "00"
    stamp = f"{ts.strftime('%Y%m%d')}_{hh}{mm}z"
    ymd   = f"Y{ts.strftime('%Y')}/M{ts.strftime('%m')}/D{ts.strftime('%d')}"

    if pd.Timestamp("2023-01-01") <= ts <= pd.Timestamp("2024-10-31 23:59:59"):
        fname = f"CF2_hindcast.{collection}.{stamp}*.nc4"
        ddir  = os.path.join(d_geoscf, "hindcast", "pub", mode, ymd)
    elif ts >= pd.Timestamp("2025-08-01"):
        fname = f"GEOS.cf.{mode}.{collection}.{stamp}*.nc4"
        ddir  = os.path.join(d_geoscf, "NRTv2", "pub", mode, ymd)
    else:
        return ""   # gap period: no data available

    matches = glob.glob(os.path.join(ddir, fname))
    return matches[0] if matches else ""


def _geoscf_interp(fpath, varname, lat, lon, lev_idx=None):
    """
    Bilinearly interpolate one variable from a GEOS-CF file to (lat, lon).

    Parameters
    ----------
    fpath    : str    full file path (returns np.nan if file missing)
    varname  : str    variable name in the NetCDF file
    lat, lon : float  target coordinates
    lev_idx  : int or None
               If the variable has a lev dimension, select this level index.
               -1 = surface (bottom) level in GEOS-CF level ordering.
               None = variable has no lev dimension (e.g. COSZ, PS).

    Returns
    -------
    float  interpolated value, or np.nan on any failure
    """
    if not os.path.isfile(fpath):
        return np.nan
    with xr.open_dataset(fpath, engine="netcdf4") as ds:
        if varname not in ds:
            return np.nan
        da = ds[varname].isel(time=0)
        if lev_idx is not None and "lev" in da.dims:
            da = da.isel(lev=lev_idx)
        val = float(da.interp(lat=lat, lon=lon, method="linear",
                              kwargs={"fill_value": "extrapolate"}).values)
    return val


# =============================================================================
# STEP 1 — PSS ratio along trajectory
# =============================================================================

def step1_pss_ratio(trop_csf, d_geoscf, suffix="_O"):
    """
    Compute the NOx/NO2 PSS ratio at every trajectory point using GEOS-CF
    fields, and append the results as suffixed columns to trop_csf.

    PSS equation (photo-stationary state):
        NOx/NO2 = 1 + J_NO2 / (k_NO+O3 × [O3])

    where:
        J_NO2  = J_NO2_overhead × max(0, COSZ)^0.7
                 (Madronich 1987 two-stream parameterisation)
        k_NO+O3 = 3.0e-12 × exp(-1500/T)
                 (JPL 19-5 Arrhenius rate constant)
        [O3]   = O3_molmol × P / (kB × T)
                 (ideal gas number density)

    All GEOS-CF fields are taken at the model surface level (lev_idx=-1),
    which is the bottom of the atmosphere — appropriate for BL plumes.

    Columns added to trop_csf (all suffixed):
        pss_ratio, jno2, k_no_o3, o3_molmol, o3_nd, T_K_geoscf, cosz, P_Pa

    Parameters
    ----------
    trop_csf  : pd.DataFrame  CSF table with columns lat_a3{s}, lon_a3{s},
                              time_tag{s}  (produced by _calc_csf)
    d_geoscf  : str           root path for GEOS-CF files (CT.d_geoscf)
    suffix    : str           column suffix, "_O" or "_H"

    Returns
    -------
    trop_csf : pd.DataFrame  with PSS columns appended in-place
    """
    s = suffix

    # Initialise output columns
    for col in ["pss_ratio", "jno2", "k_no_o3", "o3_molmol", "o3_nd",
                "T_K_geoscf", "cosz", "P_Pa"]:
        trop_csf[col + s] = np.nan

    for idx, row in trop_csf.iterrows():
        lat  = row.get(f"lat_a3{s}",  np.nan)
        lon  = row.get(f"lon_a3{s}",  np.nan)
        time = row.get(f"time_tag{s}", pd.NaT)

        if any(pd.isna(v) for v in [lat, lon, time]):
            continue

        ts = pd.Timestamp(time)

        # --- Read GEOS-CF fields at this trajectory point ---
        f_chm = _geoscf_path("chm_tavg_1hr_glo_L1440x721_slv", ts, d_geoscf)
        f_xgc = _geoscf_path("xgc_tavg_1hr_glo_L1440x721_slv", ts, d_geoscf)
        f_met = _geoscf_path("met_tavg_1hr_glo_L1440x721_slv", ts, d_geoscf)

        # O3 mole fraction at surface level [mol mol-1]
        o3_molmol = _geoscf_interp(f_chm, "O3",   lat, lon, lev_idx=-1)
        # Cosine of solar zenith angle (no lev dimension)
        cosz      = _geoscf_interp(f_xgc, "COSZ", lat, lon, lev_idx=None)
        # Temperature at surface level [K]
        T_K       = _geoscf_interp(f_met, "T",    lat, lon, lev_idx=-1)
        # Surface pressure [Pa]
        P_Pa      = _geoscf_interp(f_met, "PS",   lat, lon, lev_idx=None)
        if np.isnan(P_Pa):
            P_Pa = _P0_PA   # fallback to standard atmosphere

        # --- PSS ratio ---
        jno2    = _JNO2_OVER * max(0.0, float(cosz)) ** 0.7        # [s-1]
        k_no_o3 = _A_NO_O3 * np.exp(-_Ea_NO_O3 / T_K)             # [cm3 molec-1 s-1]
        n_air   = P_Pa / (_kB * T_K) * 1e-6                        # [molec cm-3]
        o3_nd   = o3_molmol * n_air                                 # [molec cm-3]
        denom   = k_no_o3 * o3_nd
        pss_ratio = 1.0 + jno2 / denom if denom > 0 else np.nan

        trop_csf.loc[idx, f"pss_ratio{s}"]  = pss_ratio
        trop_csf.loc[idx, f"jno2{s}"]       = jno2
        trop_csf.loc[idx, f"k_no_o3{s}"]    = k_no_o3
        trop_csf.loc[idx, f"o3_molmol{s}"]  = o3_molmol
        trop_csf.loc[idx, f"o3_nd{s}"]      = o3_nd
        trop_csf.loc[idx, f"T_K_geoscf{s}"] = T_K
        trop_csf.loc[idx, f"cosz{s}"]       = cosz
        trop_csf.loc[idx, f"P_Pa{s}"]       = P_Pa

    return trop_csf


# =============================================================================
# STEP 2 — Identify the PSS point (QC screen + log-normal plume fit)
# =============================================================================

def step2_find_pss_point(trop_csf, suffix="_O", residual_tol=2.0):
    """
    Screen CSF rows by quality flag, fit a log-normal pulse to the
    flux-vs-age distribution to identify the primary plume, and locate
    the PSS point (transport age at which flux peaks).

    The PSS point is where [NO]/[NO2] is best approximated by photo-
    stationary state — in practice, the age at which NO2 CSF is maximum,
    before dilution and chemical loss dominate.

    A log-normal pulse is fitted to flux vs. |age_hours| because the
    flux-vs-age curve is right-skewed in linear time (rises fast near the
    source, decays slowly downwind).  Outliers (e.g. flux spikes from a
    separate nearby source) are rejected by iterative MAD-based residual
    filtering before the PSS point is identified.

    Parameters
    ----------
    trop_csf     : pd.DataFrame  output of step1_pss_ratio
    suffix       : str           "_O" or "_H"
    residual_tol : float         outlier rejection threshold (MAD multiplier);
                                 lower = stricter

    Returns
    -------
    trop_csf_qc  : pd.DataFrame  QC-passed rows only
    pss_row      : pd.Series     row at the PSS point
    pss_tdump_id : int or nan    tdump_id of the PSS point
    """
    s        = suffix
    flux_col = f"flux_no2{s}"
    age_col  = f"age_hours{s}"
    qf_col   = f"QF_gauss_abs{s}"

    # --- 2a. Quality filter: positive flux + Gaussian QF passed ---
    mask = (trop_csf[flux_col].notna() & (trop_csf[flux_col] > 0)
            & (trop_csf[qf_col] == 0))
    trop_csf_qc = trop_csf.loc[mask].sort_values("tdump_id").reset_index(drop=True)

    if trop_csf_qc.empty:
        print(f"  [step2] No valid flux rows after QC (suffix={s})")
        return trop_csf_qc, pd.Series(dtype=float), np.nan

    # --- 2b. Fit log-normal pulse to flux vs. |age_hours| ---
    #         Log-normal pulse: f(t) = A * exp(-(ln(t/t_peak))^2 / (2*sigma^2))
    #         Symmetric in log-time, right-skewed in linear time.
    def _lognormal_pulse(t, A, t_peak, sigma):
        t   = np.asarray(t, dtype=float)
        out = np.zeros_like(t)
        v   = t > 0
        out[v] = A * np.exp(-(np.log(t[v] / t_peak))**2 / (2 * sigma**2))
        return out

    flux = trop_csf_qc[flux_col].values.astype(float)
    age  = np.abs(trop_csf_qc[age_col].values).astype(float)
    n    = len(flux)

    fit_ok     = False
    t_peak_fit = age[np.argmax(flux)]   # argmax fallback if fit fails
    in_primary = np.ones(n, dtype=bool)

    if n >= 4:
        i0     = np.argmax(flux)
        p0     = [flux[i0], max(age[i0], 0.1), 0.5]
        bounds = ([0, 0.01, 0.05], [flux[i0] * 5, age.max() * 1.5, 3.0])
        try:
            popt, _ = curve_fit(_lognormal_pulse, age, flux, p0=p0,
                                bounds=bounds, maxfev=5000)
            t_peak_fit = float(popt[1])
            fit_ok     = True

            # Iterative outlier rejection using MAD of residuals
            for _ in range(2):
                residuals  = flux - _lognormal_pulse(age, *popt)
                mad        = np.median(np.abs(residuals - np.median(residuals)))
                in_primary = np.abs(residuals) <= residual_tol * (mad + 1e-9)
                if in_primary.sum() >= 4:
                    try:
                        popt, _ = curve_fit(_lognormal_pulse,
                                            age[in_primary], flux[in_primary],
                                            p0=popt, bounds=bounds, maxfev=5000)
                        t_peak_fit = float(popt[1])
                    except RuntimeError:
                        break
                else:
                    in_primary = np.ones(n, dtype=bool)   # reject nothing
                    break
        except RuntimeError:
            pass   # keep argmax fallback

    trop_csf_qc["step2_in_primary_plume"] = in_primary
    n_excl = int((~in_primary).sum())
    if n_excl > 0:
        print(f"  [step2] Outlier rejection: excluded {n_excl}/{n} rows  (fit_ok={fit_ok})")

    primary = trop_csf_qc.loc[in_primary]
    if primary.empty:
        primary = trop_csf_qc   # fallback: use everything

    # --- 2c. PSS point: row closest to fitted t_peak (or argmax) ---
    if fit_ok:
        age_abs     = np.abs(primary[age_col].values)
        closest_idx = primary.index[np.argmin(np.abs(age_abs - t_peak_fit))]
        pss_row     = trop_csf_qc.loc[closest_idx]
    else:
        pss_row = primary.loc[primary[flux_col].idxmax()]

    pss_tdump_id = pss_row.get("tdump_id", np.nan)
    print(f"  [step2] PSS point: tdump_id={pss_tdump_id}  "
          f"flux_no2={pss_row[flux_col]:.4f} tNO2/hr  "
          f"age={pss_row[age_col]:.2f} hr  "
          f"t_peak_fit={t_peak_fit:.2f} hr  fit_ok={fit_ok}")

    return trop_csf_qc, pss_row, pss_tdump_id


# =============================================================================
# STEP 3 — Convert NO2 flux → NOx flux at the PSS point
# =============================================================================

def step3_no2_to_nox_flux(pss_row, suffix="_O"):
    """
    Convert NO2 mass flux at the PSS point to NOx mass flux using the PSS
    molar ratio and an effective NOx molecular weight.

    At PSS:  NOx/NO2 = pss_ratio  (molar)
    NO mole fraction in NOx:  f_NO = (pss_ratio - 1) / pss_ratio
    Effective NOx MW:  MW_NOx = f_NO * MW_NO + (1 - f_NO) * MW_NO2

    Mass flux conversion:
        flux_nox = flux_no2 * pss_ratio * (MW_NOx / MW_NO2)

    Parameters
    ----------
    pss_row : pd.Series  PSS point row from step2_find_pss_point
    suffix  : str        "_O" or "_H"

    Returns
    -------
    flux_nox_PSS : float  [tNOx/hr]
    pss_ratio    : float  NOx/NO2 molar ratio at PSS
    """
    s         = suffix
    flux_no2  = float(pss_row.get(f"flux_no2{s}",  np.nan))
    pss_ratio = float(pss_row.get(f"pss_ratio{s}", np.nan))

    if np.isnan(flux_no2) or np.isnan(pss_ratio):
        print(f"  [step3] Missing flux_no2 or pss_ratio — aborting.")
        return np.nan, np.nan

    f_NO         = (pss_ratio - 1.0) / pss_ratio
    MW_NOx_eff   = f_NO * _MW_NO + (1.0 - f_NO) * _MW_NO2
    flux_nox_PSS = flux_no2 * pss_ratio * (MW_NOx_eff / _MW_NO2)

    print(f"  [step3] flux_no2={flux_no2:.4f} tNO2/hr  pss_ratio={pss_ratio:.3f}  "
          f"f_NO={f_NO:.3f}  MW_NOx={MW_NOx_eff:.2f} g/mol  "
          f"→ flux_nox_PSS={flux_nox_PSS:.4f} tNOx/hr")

    return flux_nox_PSS, pss_ratio


# =============================================================================
# STEP 4 — Decay correction: back-extrapolate NOx flux to t=0 (source)
# =============================================================================

def step4_decay_correction(trop_csf_qc, pss_row, flux_nox_PSS,
                            d_geoscf, suffix="_O", option="B"):
    """
    Back-extrapolate NOx flux from the PSS point to t=0 (the emission source)
    by correcting for chemical decay during transport.

    Two options:

    Option A — Empirical
        Fit flux_nox(t) = flux_nox_PSS × exp(-k_eff × (t - t_PSS)) to the
        post-PSS rows, where flux_nox = flux_no2 × pss_ratio.  Then
        extrapolate back to t=0:
            flux_nox_source = flux_nox_PSS × exp(+k_eff × t_PSS)

    Option B — Analytical  (default)
        Compute k_loss from GEOS-CF boundary-layer [OH] at the PSS point
        using the JPL 2019 termolecular rate for NO2 + OH + M → HNO3.
        Then:
            flux_nox_source = flux_nox_PSS × exp(+k_loss × t_PSS)
        [OH] and T are taken at the model surface level (lev_idx=-1),
        consistent with the boundary-layer M used in the rate constant.

    Parameters
    ----------
    trop_csf_qc  : pd.DataFrame  QC-passed rows from step2
    pss_row      : pd.Series     PSS point row from step2
    flux_nox_PSS : float         NOx flux at PSS [tNOx/hr]
    d_geoscf     : str           CT.d_geoscf
    suffix       : str           "_O" or "_H"
    option       : str           "A" or "B"

    Returns
    -------
    dict with keys:
        flux_nox_source : float  [tNOx/hr]
        flux_nox_fit    : pd.Series or None  per-row fitted decay curve
        k_loss          : float  [s-1]  (Option B; nan for A)
        k_eff           : float  [s-1]  (Option A; nan for B)
        age_at_PSS_s    : float  [s]
        oh_nd           : float  [molec cm-3]  (Option B; nan for A)
        option          : str
    """
    s       = suffix
    age_col = f"age_hours{s}"

    age_pss_s = abs(float(pss_row.get(age_col, np.nan))) * 3600.0

    result = {"flux_nox_source": np.nan, "flux_nox_fit": None,
              "k_loss": np.nan, "k_eff": np.nan,
              "age_at_PSS_s": age_pss_s, "oh_nd": np.nan, "option": option}

    if np.isnan(flux_nox_PSS) or np.isnan(age_pss_s):
        print("  [step4] Missing flux_nox_PSS or age_at_PSS — skipping.")
        return result

    # Fitted decay curve evaluated at every QC row (for plotting)
    def _fit_curve(k):
        t_s = np.abs(trop_csf_qc[age_col].values) * 3600.0
        return pd.Series(flux_nox_PSS * np.exp(-k * (t_s - age_pss_s)),
                         index=trop_csf_qc.index)

    # --- Option A: empirical exponential fit to post-PSS flux decline ---
    if option == "A":
        pss_tid  = pss_row.get("tdump_id", -1)
        flux_col = f"flux_no2{s}"
        pss_ratio_col = f"pss_ratio{s}"

        post = trop_csf_qc.loc[trop_csf_qc["tdump_id"] > pss_tid].copy()
        if flux_col not in post.columns or pss_ratio_col not in post.columns:
            print("  [step4-A] Required columns missing.")
            return result

        post["flux_nox"] = post[flux_col] * post[pss_ratio_col]
        post = post.dropna(subset=[age_col, "flux_nox"])
        post = post.loc[post["flux_nox"] > 0]

        if len(post) < 3:
            print(f"  [step4-A] Insufficient post-PSS points (n={len(post)}).")
            return result

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
                       "k_eff": k_eff, "flux_nox_fit": _fit_curve(k_eff)})
        print(f"  [step4-A] k_eff={k_eff:.2e} s-1  "
              f"age_PSS={age_pss_s/3600:.2f} hr  "
              f"flux_nox_source={flux_nox_source:.4f} tNOx/hr")

    # --- Option B: analytical k_loss from GEOS-CF boundary-layer [OH] ---
    elif option == "B":
        lat  = float(pss_row.get(f"lat_a3{s}",   np.nan))
        lon  = float(pss_row.get(f"lon_a3{s}",   np.nan))
        time = pss_row.get(f"time_tag{s}", pd.NaT)
        T_K  = float(pss_row.get(f"T_K_geoscf{s}", np.nan))

        if any(pd.isna(v) for v in [lat, lon, time]):
            print("  [step4-B] Missing lat/lon/time for OH look-up.")
            return result

        ts    = pd.Timestamp(time)
        f_oxi = _geoscf_path("oxi_inst_1hr_glo_L1440x721_v72", ts, d_geoscf, mode="ana")
        f_met = _geoscf_path("met_inst_1hr_glo_L1440x721_v72", ts, d_geoscf, mode="ana")

        # OH and T at surface (boundary layer) level; consistent with _M_BL
        oh_molmol = _geoscf_interp(f_oxi, "OH", lat, lon, lev_idx=-1)  # [mol mol-1]
        T_sfc     = _geoscf_interp(f_met, "T",  lat, lon, lev_idx=-1)  # [K]
        P_sfc     = _geoscf_interp(f_met, "PS", lat, lon, lev_idx=None) # [Pa]
        if np.isnan(T_sfc):
            T_sfc = T_K if not np.isnan(T_K) else 298.0
        if np.isnan(P_sfc):
            P_sfc = _P0_PA

        # [OH] number density [molec cm-3]
        n_air = P_sfc / (_kB * T_sfc) * 1e-6
        oh_nd = oh_molmol * n_air

        # JPL 2019 termolecular rate for NO2 + OH + M → HNO3
        Tr      = T_sfc / 300.0
        k0      = 2.4e-30 * Tr**(-3.0) * _M_BL
        kinf    = 1.7e-11 * Tr**(-2.1)
        k_f     = k0 / (1.0 + k0 / kinf)
        N       = 1.0 + (np.log10(k0 / kinf))**2
        k_NO2OH = k_f * 0.6**(1.0 / N)    # [cm3 molec-1 s-1]  (Fc=0.6)
        k_loss  = k_NO2OH * oh_nd          # [s-1]

        flux_nox_source = flux_nox_PSS * np.exp(k_loss * age_pss_s)
        result.update({"flux_nox_source": flux_nox_source,
                       "k_loss": k_loss, "oh_nd": oh_nd,
                       "flux_nox_fit": _fit_curve(k_loss)})
        print(f"  [step4-B] [OH]={oh_nd:.2e} molec/cm3  "
              f"k_loss={k_loss:.2e} s-1  "
              f"age_PSS={age_pss_s/3600:.2f} hr  "
              f"flux_nox_source={flux_nox_source:.4f} tNOx/hr")

    return result


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_nox_workflow(trop_csf, true_tdump, d_geoscf,
                     suffix="_O", option_b_or_a="B", residual_tol=2.0):
    """
    Orchestrate the five-step PSS NOx back-calculation.

    Call from _csf_prcs() after the optimised CSF (_O) has been computed:

        trop_csf, nox_result = run_nox_workflow(
            trop_csf, true_tdump, CT.d_geoscf, suffix="_O", option_b_or_a="B")

    Columns added to trop_csf
    -------------------------
    Step 1 (per row, suffixed):
        pss_ratio, jno2, k_no_o3, o3_molmol, o3_nd, T_K_geoscf, cosz, P_Pa
    Step 2 (per row, suffixed):
        step2_qc_pass, step2_is_pss_point
    Step 4 (per row, suffixed):
        flux_nox_fit

    Parameters
    ----------
    trop_csf      : pd.DataFrame  merged CSF table (_H and _O columns)
    true_tdump    : pd.DataFrame  optimised trajectory from _TDOPT_combined
    d_geoscf      : str           CT.d_geoscf
    suffix        : str           which trajectory to use ("_O" or "_H")
    option_b_or_a : str           decay correction method ("B" or "A")
    residual_tol  : float         step2 outlier rejection threshold

    Returns
    -------
    trop_csf   : pd.DataFrame
    nox_result : dict  keys: status, pss_tdump_id, pss_ratio, flux_nox_PSS,
                             flux_nox_source, k_loss, k_eff, age_at_PSS_s,
                             oh_nd, option
    """
    print("  [NOx workflow] Starting 5-step PSS back-calculation ...")

    # Step 1 — PSS ratio at every trajectory point
    trop_csf = step1_pss_ratio(trop_csf, d_geoscf, suffix=suffix)

    # Step 2 — QC screen + locate PSS point
    trop_csf_qc, pss_row, pss_tdump_id = step2_find_pss_point(
        trop_csf, suffix=suffix, residual_tol=residual_tol)
    trop_csf[f"step2_qc_pass{suffix}"]     = trop_csf.index.isin(trop_csf_qc.index)
    trop_csf[f"step2_is_pss_point{suffix}"] = trop_csf["tdump_id"] == pss_tdump_id

    if pss_row.empty or np.isnan(pss_tdump_id):
        print("  [NOx workflow] No valid PSS point — aborting.")
        return trop_csf, {"status": "no_pss_point"}

    # Step 3 — NO2 flux → NOx flux at PSS
    flux_nox_PSS, pss_ratio = step3_no2_to_nox_flux(pss_row, suffix=suffix)
    if np.isnan(flux_nox_PSS):
        print("  [NOx workflow] Step 3 failed — aborting.")
        return trop_csf, {"status": "step3_failed"}

    # Step 4 — decay correction back to t=0
    decay = step4_decay_correction(
        trop_csf_qc, pss_row, flux_nox_PSS,
        d_geoscf, suffix=suffix, option=option_b_or_a)

    # Write per-row fitted decay curve into trop_csf
    trop_csf[f"flux_nox_fit{suffix}"] = np.nan
    if decay["flux_nox_fit"] is not None:
        trop_csf.loc[decay["flux_nox_fit"].index,
                     f"flux_nox_fit{suffix}"] = decay["flux_nox_fit"].values

    # Step 5 — collect scalar results
    nox_result = {
        "status":       "ok",
        "pss_tdump_id": pss_tdump_id,
        "pss_ratio":    pss_ratio,
        "flux_nox_PSS": flux_nox_PSS,
        **{k: v for k, v in decay.items() if k != "flux_nox_fit"},
    }
    print("  [NOx workflow] Complete.")
    return trop_csf, nox_result

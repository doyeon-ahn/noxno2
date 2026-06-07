# =============================================================================
# csf_func_tdopt.py  —  Optimised plume trajectory for CSF processing
# =============================================================================
# Constructs a smooth, single-plume trajectory from TROPOMI NO2 pixels using
# a four-step pipeline:
#
#   Step 1.  Filter a3 outliers
#            Reject erroneous Gaussian peak positions (lon_a3_H, lat_a3_H)
#            using local directional consistency.  a3 outliers arise when the
#            HYSPLIT trajectory deviates from the true wind direction (more
#            likely at longer transport ages) or when a secondary plume
#            dominates a transect.  Detected by the jump-and-return signature:
#            large turn at point i followed by a compensating turn at i+1,
#            and/or anomalously large step distance to point i.
#
#   Step 2.  Build SNR raster
#            Compute a 3x3 neighbourhood SNR for every pixel in the native
#            TROPOMI scanline x ground_pixel raster.  Pixels above
#            snr_critical_pct are plume candidates.
#
#   Step 3.  BFS flood-fill seeded from filtered a3 anchors
#            Grow a connected plume mask from the inlier a3 positions.
#            Seeding from a3 anchors (rather than the facility) keeps the
#            flood-fill within the single plume already identified by the
#            Gaussian fit, preventing bleed-over into adjacent plumes.
#
#   Step 4.  Dijkstra highest-NO2 spine
#            Extract the globally optimal path through the plume mask
#            (edge cost = 1/NO2), from the source-end anchor to the
#            physically farthest reachable pixel.
#
#   Step 5.  Polynomial smoothing
#            Fit lon(age_km) and lat(age_km) polynomials through the raw
#            Dijkstra spine to remove pixel-scale raster noise, then
#            recompute wind fields on the smoothed geometry.
#
# Public entry point:  _TDOPT_combined(trop_csf_H, trop, tdump, ...)
# =============================================================================

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


# =============================================================================
# GEOMETRY HELPERS  (local copies — no dependency on csf_prcs.CFG)
# =============================================================================

def _haversine_m(lat1, lon1, lat2, lon2, R=6_371_000.0):
    """Vectorised haversine distance [m]."""
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a    = (np.sin(dlat / 2)**2
            + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2))
            * np.sin(dlon / 2)**2)
    return R * 2 * np.arcsin(np.sqrt(a))


def _calculate_bearing(lat1, lon1, lat2, lon2):
    """Bearing from point-1 → point-2 [degrees, 0–360)."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    y = np.sin(lon2 - lon1) * np.cos(lat2)
    x = (np.cos(lat1) * np.sin(lat2)
         - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1))
    return (np.degrees(np.arctan2(y, x)) + 360) % 360


# =============================================================================
# STEP 1 — Filter erroneous a3 positions
# =============================================================================

def _filter_a3_outliers(qc_rows_H, turn_thresh_deg=45., step_mad_k=3.0):
    """
    Flag erroneous a3 positions using local directional consistency.

    Two independent criteria are applied; a point is flagged as an outlier
    if either fires.

    Criterion 1 — jump-and-return turn angle
        For each interior point i, compute the signed turn angle from step
        (i-1 → i) to step (i → i+1).  Flag point i if:
            |turn_i|    > turn_thresh_deg   (large jump at i)
            |turn_i+1|  > turn_thresh_deg   (large return at i+1)
            sign(turn_i) != sign(turn_i+1)  (turns are compensating)
        This distinguishes transient spikes from real persistent wind changes:
        a real wind-direction shift produces a large turn at one point but
        small turns afterward; a spike produces two large opposite turns.

    Criterion 2 — anomalous step distance
        The haversine distance between consecutive a3 positions should be
        roughly constant (wind_speed × 15 min).  Flag point i if its
        incoming step distance exceeds median + step_mad_k × MAD.

    Parameters
    ----------
    qc_rows_H      : pd.DataFrame  QF_gauss_abs_H==0 rows from trop_csf_H,
                                   sorted by age_hours_H;
                                   must contain lon_a3_H, lat_a3_H, age_hours_H
    turn_thresh_deg: float         turn angle threshold [deg]; 45° allows
                                   normal wind veering at 15-min a3 spacing
    step_mad_k     : float         step distance outlier threshold [MAD units]

    Returns
    -------
    inlier : np.ndarray bool  shape (N,)  True = trusted a3 anchor
    """
    df   = qc_rows_H.sort_values("age_hours_H").reset_index(drop=True)
    N    = len(df)
    lons = df["lon_a3_H"].values
    lats = df["lat_a3_H"].values

    inlier = np.ones(N, dtype=bool)
    if N < 3:
        return inlier   # too few points to assess consistency

    # --- Criterion 1: jump-and-return turn angle -------------------------

    def _signed_turn(b1, b2):
        """Signed angular change from bearing b1 to b2, in [-180, 180]."""
        return ((b2 - b1 + 180.) % 360.) - 180.

    # bearing of each consecutive a3 step, shape (N-1,)
    bearings = np.array([
        _calculate_bearing(lats[k], lons[k], lats[k+1], lons[k+1])
        for k in range(N - 1)
    ])

    # signed turn angle at each interior point, shape (N-2,)
    turns = np.array([
        _signed_turn(bearings[k], bearings[k+1])
        for k in range(N - 2)
    ])

    for i in range(1, N - 1):
        turn_i = turns[i - 1]
        if i < N - 2:
            turn_next = turns[i]
            # jump-and-return: both turns large and opposite in sign
            if (abs(turn_i)    > turn_thresh_deg and
                abs(turn_next) > turn_thresh_deg and
                np.sign(turn_i) != np.sign(turn_next)):
                inlier[i] = False
        else:
            # last interior point: no following turn available;
            # flag on magnitude alone to avoid anchoring the far end badly
            if abs(turn_i) > turn_thresh_deg:
                inlier[i] = False

    # --- Criterion 2: anomalous step distance ----------------------------

    steps = np.array([
        _haversine_m(lats[k], lons[k], lats[k+1], lons[k+1])
        for k in range(N - 1)
    ])  # shape (N-1,)

    step_median = np.median(steps)
    step_mad    = np.median(np.abs(steps - step_median))
    step_thresh = step_median + step_mad_k * step_mad

    for i in range(1, N):
        if steps[i - 1] > step_thresh:
            inlier[i] = False

    n_rejected = int((~inlier).sum())
    if n_rejected > 0:
        print(f"  [TDOPT] a3 outlier filter: rejected {n_rejected}/{N} anchors")

    return inlier


# =============================================================================
# STEP 2 — SNR raster
# =============================================================================

def _build_snr_raster(sls, gps, no2, sig):
    """
    Compute a 2-D SNR raster on the native TROPOMI scanline x ground_pixel grid.

    SNR at each pixel is computed from its 3x3 neighbourhood:
        signal = mean(patch) - scene_background   (scene_background = median)
        noise  = sqrt(instrument_variance + background_variance)
    where background_variance is the scene IQR.

    Pixels with fewer than 6 valid neighbours receive NaN SNR.

    Parameters
    ----------
    sls : np.ndarray int    scanline indices,     shape (N,)
    gps : np.ndarray int    ground_pixel indices, shape (N,)
    no2 : np.ndarray float  NO2 values [µmol m-2], shape (N,)
    sig : np.ndarray float  NO2 precision,         shape (N,)

    Returns
    -------
    snr_2d  : np.ndarray  shape (nsl, ngp)
    zbg     : float       scene background NO2
    zbg_sig : float       background variability (IQR)
    """
    nsl = int(sls.max()) + 1
    ngp = int(gps.max()) + 1

    no2_2d = np.full((nsl, ngp), np.nan)
    sig_2d = np.full((nsl, ngp), np.nan)
    for k in range(len(no2)):
        no2_2d[sls[k], gps[k]] = no2[k]
        sig_2d[sls[k], gps[k]] = sig[k]

    zbg     = np.nanmedian(no2_2d)
    zbg_sig = np.nanquantile(no2_2d, 0.75) - np.nanquantile(no2_2d, 0.25)

    snr_2d = np.full((nsl, ngp), np.nan)
    for i in range(1, nsl - 1):
        for j in range(1, ngp - 1):
            patch   = no2_2d[i-1:i+2, j-1:j+2]
            psig    = sig_2d[i-1:i+2, j-1:j+2]
            n_valid = np.sum(~np.isnan(patch))
            if n_valid < 6:
                continue
            zobs           = np.nanmean(patch)
            sig_instrument = np.nansum(psig**2)**0.5 / n_valid
            snr_2d[i, j]   = (zobs - zbg) / (sig_instrument**2 + zbg_sig**2)**0.5

    return snr_2d, zbg, zbg_sig


# =============================================================================
# STEP 3 — BFS flood-fill seeded from filtered a3 anchors
# =============================================================================

def _flood_fill_from_anchors(sls, gps, lons, lats, no2, snr_2d,
                              snr_critical_pct, qc_rows_H_filtered):
    """
    Grow a connected plume mask in the TROPOMI raster, seeded from the
    filtered a3 anchor pixels.

    Candidate plume pixels are those above snr_critical_pct of the SNR
    distribution.  Seeds are force-included even if below the SNR threshold
    (the Gaussian fit already confirmed a plume centre there).  BFS expansion
    is 8-connected (Moore neighbourhood).

    Seeding from filtered a3 anchors — rather than the facility location —
    keeps the flood-fill within the single plume identified by the Gaussian
    fit and prevents bleed-over into adjacent unrelated plumes.

    Parameters
    ----------
    sls, gps             : np.ndarray int    scanline / ground_pixel indices
    lons, lats           : np.ndarray float  pixel coordinates
    no2                  : np.ndarray float  NO2 values
    snr_2d               : np.ndarray        SNR raster from _build_snr_raster
    snr_critical_pct     : float             percentile threshold (e.g. 0.75)
    qc_rows_H_filtered   : pd.DataFrame      inlier QC rows after a3 outlier
                                             filtering; must contain
                                             lon_a3_H, lat_a3_H

    Returns
    -------
    plume_2d  : np.ndarray  shape (nsl, ngp)
                1.0 = confirmed plume pixel, NaN = rejected
    seed_cells: set of (sl, gp) tuples used as flood-fill seeds
    """
    nsl = snr_2d.shape[0]
    ngp = snr_2d.shape[1]
    snr_critical = np.nanquantile(snr_2d, snr_critical_pct)

    # Candidate pixels above SNR threshold: mark as unconfirmed (-9999)
    plume_2d = np.where(snr_2d >= snr_critical, -9999., np.nan)

    # Map each filtered a3 anchor to the nearest TROPOMI pixel and seed it
    seed_cells = set()
    for _, row in qc_rows_H_filtered.iterrows():
        dist2  = (lons - row["lon_a3_H"])**2 + (lats - row["lat_a3_H"])**2
        k0     = int(np.argmin(dist2))
        si, sj = int(sls[k0]), int(gps[k0])
        plume_2d[si, sj] = -9999.   # force-include even if SNR below threshold
        seed_cells.add((si, sj))

    # Confirm all seeds
    for (si, sj) in seed_cells:
        plume_2d[si, sj] = 1.0

    # BFS: expand confirmed pixels into adjacent unconfirmed candidates
    processed = np.zeros_like(plume_2d)
    while True:
        ri, ci = np.where((plume_2d == 1.) & (processed == 0.))
        if len(ri) == 0:
            break
        processed[ri, ci] = 1.
        for k in range(len(ri)):
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    ni, nj = ri[k] + di, ci[k] + dj
                    if 0 <= ni < nsl and 0 <= nj < ngp:
                        if plume_2d[ni, nj] == -9999.:
                            plume_2d[ni, nj] = 1.

    return plume_2d, seed_cells


# =============================================================================
# STEP 4 — Dijkstra highest-NO2 spine
# =============================================================================

def _dijkstra_spine(sls, gps, lons, lats, no2, plume_2d, qc_rows_H_filtered):
    """
    Extract the globally optimal highest-NO2 spine through the plume mask
    using Dijkstra's algorithm.

    Edge cost = 1 / mean(NO2) of the two endpoint pixels, so Dijkstra
    minimises cost by preferentially routing through high-NO2 pixels.

    Source  : the confirmed pixel nearest to the youngest (smallest
              |age_hours_H|) filtered a3 anchor — i.e., the source end.
    Endpoint: the physically farthest reachable plume pixel from the source.

    Parameters
    ----------
    sls, gps             : np.ndarray int
    lons, lats           : np.ndarray float
    no2                  : np.ndarray float
    plume_2d             : np.ndarray  flood-fill output (1.0 = plume pixel)
    qc_rows_H_filtered   : pd.DataFrame  inlier QC rows; must contain
                                         age_hours_H, lon_a3_H, lat_a3_H

    Returns
    -------
    waypoints : list of (sl, gp) tuples along the optimal spine,
                source-end first; empty list if fewer than 2 plume pixels
    grid      : dict  (sl, gp) → no2
    cell_lon  : dict  (sl, gp) → longitude
    cell_lat  : dict  (sl, gp) → latitude
    """
    # Build pixel lookup restricted to confirmed plume pixels
    grid     = {}
    cell_lon = {}
    cell_lat = {}
    for k in range(len(no2)):
        key = (int(sls[k]), int(gps[k]))
        if plume_2d[int(sls[k]), int(gps[k])] == 1.:
            grid[key]     = no2[k]
            cell_lon[key] = lons[k]
            cell_lat[key] = lats[k]

    plume_keys = list(grid.keys())
    if len(plume_keys) < 2:
        return [], grid, cell_lon, cell_lat

    key_to_idx = {k: i for i, k in enumerate(plume_keys)}
    N = len(plume_keys)

    # Build sparse adjacency matrix: low cost = high NO2
    rows_sp, cols_sp, weights = [], [], []
    for key in plume_keys:
        si, sj = key
        i = key_to_idx[key]
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                nb = (si + di, sj + dj)
                if nb in key_to_idx:
                    j    = key_to_idx[nb]
                    cost = 1.0 / (0.5 * (grid[key] + grid[nb]) + 1e-9)
                    rows_sp.append(i);  cols_sp.append(j);  weights.append(cost)

    graph = csr_matrix((weights, (rows_sp, cols_sp)), shape=(N, N))

    # Source = filtered a3 anchor with smallest |age_hours_H| (nearest source)
    youngest   = qc_rows_H_filtered.loc[
        qc_rows_H_filtered["age_hours_H"].abs().idxmin()]
    dist2_seed = ((lons - youngest["lon_a3_H"])**2
                  + (lats - youngest["lat_a3_H"])**2)
    k_seed     = int(np.argmin(dist2_seed))
    seed_key   = (int(sls[k_seed]), int(gps[k_seed]))
    if seed_key not in key_to_idx:
        seed_key = plume_keys[0]   # fallback if seed pixel not in plume mask
    seed_idx = key_to_idx[seed_key]

    dist_arr, predecessors = dijkstra(
        graph, indices=seed_idx, return_predecessors=True)

    # Endpoint = physically farthest reachable plume pixel from the source
    seed_lon  = cell_lon[seed_key]
    seed_lat  = cell_lat[seed_key]
    phys_dist = np.array([
        _haversine_m(seed_lat, seed_lon,
                     cell_lat[plume_keys[i]], cell_lon[plume_keys[i]])
        if not np.isinf(dist_arr[i]) else -np.inf
        for i in range(N)
    ])
    end_idx = int(np.argmax(phys_dist))

    # Traceback from endpoint to seed, then reverse to source-first order
    path = []
    cur  = end_idx
    while cur != seed_idx and cur >= 0:
        path.append(cur)
        cur = predecessors[cur]
    path.append(seed_idx)
    path.reverse()

    waypoints = [plume_keys[i] for i in path]
    return waypoints, grid, cell_lon, cell_lat


# =============================================================================
# STEP 5 — Assemble spine DataFrame + polynomial smoothing
# =============================================================================

def _assemble_and_smooth(waypoints, grid, cell_lon, cell_lat, tdump, poly_deg):
    """
    Assemble a raw spine DataFrame from Dijkstra waypoints, compute cumulative
    along-spine distance and transport age, then smooth the geometry with a
    polynomial fit and recompute wind fields.

    Parameters
    ----------
    waypoints    : list of (sl, gp) tuples  (source-end first)
    grid         : dict  (sl, gp) → no2
    cell_lon/lat : dict  (sl, gp) → coordinate
    tdump        : pd.DataFrame  HYSPLIT trajectory (for dt and scene-mean wso)
    poly_deg     : int           polynomial degree for lon/lat smoothing

    Returns
    -------
    chain : pd.DataFrame  smoothed spine; columns: tdump_id, longitude,
            latitude, no2, age_km, age_hours, wso, wd
    """
    rows  = [{"longitude": cell_lon[k], "latitude": cell_lat[k], "no2": grid[k]}
             for k in waypoints]
    chain = pd.DataFrame(rows).reset_index(drop=True)

    # Cumulative along-spine distance and transport age
    clats  = chain["latitude"].values
    clons  = chain["longitude"].values
    step_m = np.zeros(len(clats))
    step_m[1:] = _haversine_m(clats[:-1], clons[:-1], clats[1:], clons[1:])
    chain["age_km"] = np.cumsum(step_m) / 1000.0

    wso_scene          = float(tdump["wso"].replace(-9999., np.nan).mean())
    chain["wso"]       = wso_scene
    chain["age_hours"] = chain["age_km"] / (wso_scene * 3.6 + 1e-9)

    bearings    = _calculate_bearing(clats[0], clons[0], clats, clons)
    bearings[0] = bearings[1]
    chain["wd"]       = bearings
    chain["tdump_id"] = chain.index

    # Polynomial smoothing of lon/lat along the spine
    chain = chain.drop_duplicates("age_km").reset_index(drop=True)
    if len(chain) >= poly_deg + 1:
        t   = chain["age_km"].values
        t_n = (t - t[0]) / (t[-1] - t[0] + 1e-9)   # normalise to [0,1]

        plon = np.polyfit(t_n, chain["longitude"].values, poly_deg)
        plat = np.polyfit(t_n, chain["latitude"].values,  poly_deg)
        chain["longitude"] = np.polyval(plon, t_n)
        chain["latitude"]  = np.polyval(plat, t_n)

        # Recompute all geometry-derived fields on the smoothed coordinates
        clats  = chain["latitude"].values
        clons  = chain["longitude"].values
        step_m = np.zeros(len(clats))
        step_m[1:] = _haversine_m(clats[:-1], clons[:-1], clats[1:], clons[1:])
        chain["age_km"] = np.cumsum(step_m) / 1000.0

        dt       = (tdump.loc[1, "tstmp"] - tdump.loc[0, "tstmp"]).seconds
        wso_step = _haversine_m(clats[:-1], clons[:-1], clats[1:], clons[1:]) / dt
        chain["wso"]       = np.append(wso_step, wso_scene)   # last row: scene mean
        chain["age_hours"] = chain["age_km"] / (wso_scene * 3.6 + 1e-9)

        bearings    = _calculate_bearing(clats[0], clons[0], clats, clons)
        bearings[0] = bearings[1]
        chain["wd"]       = bearings
        chain["tdump_id"] = chain.index

    return chain


# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================

def _TDOPT_combined(trop_csf_H, trop, tdump,
                    snr_critical_pct=0.75, poly_deg=2,
                    turn_thresh_deg=45., step_mad_k=3.0):
    """
    Build a smooth, single-plume optimised trajectory from TROPOMI NO2 pixels.

    Pipeline
    --------
    Step 1  _filter_a3_outliers       reject erroneous a3 positions
    Step 2  _build_snr_raster         identify plume candidate pixels by SNR
    Step 3  _flood_fill_from_anchors  grow connected plume mask from a3 seeds
    Step 4  _dijkstra_spine           extract highest-NO2 spine through mask
    Step 5  _assemble_and_smooth      build trajectory DataFrame + smooth

    Parameters
    ----------
    trop_csf_H       : pd.DataFrame  HYSPLIT CSF output (columns suffixed _H);
                                     must contain QF_gauss_abs_H, lon_a3_H,
                                     lat_a3_H, age_hours_H
    trop             : pd.DataFrame  TROPOMI pixel table; must contain
                                     scanline, ground_pixel, no2,
                                     no2_precision, longitude, latitude
    tdump            : pd.DataFrame  HYSPLIT trajectory; must contain wso, tstmp
    snr_critical_pct : float         SNR percentile for plume pixel selection
                                     (default 0.75 → top 25% SNR pixels)
    poly_deg         : int           polynomial degree for spine smoothing
                                     (default 2; increase for longer plumes)
    turn_thresh_deg  : float         a3 outlier turn angle threshold [deg]
                                     (default 45°)
    step_mad_k       : float         a3 outlier step distance threshold
                                     [MAD units] (default 3.0)

    Returns
    -------
    true_tdump : pd.DataFrame  smoothed plume spine; empty if pipeline fails
    plume_lons : np.ndarray    longitudes of all flood-fill plume pixels
    plume_lats : np.ndarray    latitudes  of all flood-fill plume pixels
    """
    sls  = trop["scanline"].values.astype(int)
    gps  = trop["ground_pixel"].values.astype(int)
    lons = trop["longitude"].values
    lats = trop["latitude"].values
    no2  = trop["no2"].values
    sig  = trop["no2_precision"].values

    # --- Guard: need at least some QC-passed HYSPLIT transects -----------
    qc_rows_H = trop_csf_H.loc[trop_csf_H["QF_gauss_abs_H"] == 0].copy()
    if len(qc_rows_H) == 0:
        print("  [TDOPT] No QC-passed transects in HYSPLIT CSF — aborting.")
        return pd.DataFrame(), np.array([]), np.array([])

    # Step 1 — reject erroneous a3 positions
    inlier            = _filter_a3_outliers(qc_rows_H, turn_thresh_deg, step_mad_k)
    qc_rows_filtered  = qc_rows_H.iloc[inlier].reset_index(drop=True)
    if len(qc_rows_filtered) == 0:
        print("  [TDOPT] All a3 anchors rejected — falling back to full QC set.")
        qc_rows_filtered = qc_rows_H   # better than empty

    # Step 2 — SNR raster
    snr_2d, _, _ = _build_snr_raster(sls, gps, no2, sig)

    # Step 3 — BFS flood-fill from filtered a3 anchors
    plume_2d, seed_cells = _flood_fill_from_anchors(
        sls, gps, lons, lats, no2, snr_2d, snr_critical_pct, qc_rows_filtered)

    # Step 4 — Dijkstra highest-NO2 spine
    waypoints, grid, cell_lon, cell_lat = _dijkstra_spine(
        sls, gps, lons, lats, no2, plume_2d, qc_rows_filtered)

    if len(waypoints) < 2:
        print("  [TDOPT] Dijkstra spine has fewer than 2 waypoints — aborting.")
        return pd.DataFrame(), np.array([]), np.array([])

    # Step 5 — assemble + smooth
    true_tdump = _assemble_and_smooth(
        waypoints, grid, cell_lon, cell_lat, tdump, poly_deg)

    # Plume pixel coordinates for map overlay
    key_to_idx = {k: i for i, k in enumerate(list(grid.keys()))}
    plume_mask = np.array([(int(sls[k]), int(gps[k])) in key_to_idx
                           for k in range(len(no2))])
    plume_lons = lons[plume_mask]
    plume_lats = lats[plume_mask]

    print(f"  [TDOPT] spine: {len(true_tdump)} pts  "
          f"{true_tdump['age_km'].max():.1f} km  "
          f"plume mask: {plume_mask.sum()} pixels  "
          f"(a3 anchors used: {len(qc_rows_filtered)}/{len(qc_rows_H)})")

    return true_tdump, plume_lons, plume_lats

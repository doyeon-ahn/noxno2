# =============================================================================
# 2_plot_map_pace_tropomi.py
# Side-by-side map comparison: PACE NO2  vs  TROPOMI NO2
# One figure per co-located pair in pace_filter_summary_2167.csv
# Pixels rendered as filled polygons (true corners for TROPOMI,
# centre-spacing-derived corners for PACE).
# =============================================================================
import os, sys
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib as mpl
mpl.use('Agg', force=True)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import PolyCollection
import contextily as ctx
import numpy as np
import pandas as pd
import xarray as xr
sys.path.insert(0, os.pardir)
import CT

## --------------------------------------------------------------------------
## PARAMETERS
## --------------------------------------------------------------------------
CFG = {
    'f_summary':    CT.d_noxno2 + '/d_dat/pace/' + CT.PRCS_VER['pace'] + '/d_nc/2167_New_Madrid_Power_Plant/pace_filter_summary_2167.csv',
    'd_pace_nc':    CT.d_noxno2 + '/d_dat/pace/' + CT.PRCS_VER['pace'] + '/d_nc/2167_New_Madrid_Power_Plant/',
    'd_trop_nc':    CT.d_noxno2 + '/d_dat/trop/' + CT.PRCS_VER['trop'] + '/d_nc/2167_New_Madrid_Power_Plant/',
    'tropomi_csv':  CT.d_noxno2 + '/d_dat/trop/' + CT.PRCS_VER['trop'] + '/d_csv/' + '2167_New_Madrid_Power_Plant_' + CT.PRCS_VER['trop'] + '.csv',
    'd_out':        CT.d_noxno2 + '/d_fig/pace_tropomi_maps/2167_New_Madrid_Power_Plant/',
    'map_buffer':   2.5,
    'trop_qa_min':  0.5,
    'cmap':         'viridis',
    'pct_lo':       0.02,
    'pct_hi':       0.98,
    'dt_max_hr':    6.0,
    'facility_id':  2167,
}

_cems           = pd.read_csv(CT.df_target)
_target         = _cems.loc[_cems['facilityId'] == CFG['facility_id']].iloc[0]
CFG['target_lat']  = float(_target['lat'])
CFG['target_lon']  = float(_target['lon'])

## --------------------------------------------------------------------------
## PIXEL POLYGON BUILDERS
## --------------------------------------------------------------------------

def _trop_polygons(fpath, ver, qa_min, lat_c, lon_c, buf):
    """
    Load TROPOMI pixel polygons from latitude_bounds / longitude_bounds (4 corners).
    Returns (verts, values) arrays or (None, None).
    verts : (N, 4, 2)  [lon, lat] per corner
    values: (N,)       µmol/m²
    """
    suffix = f'_2167_New_Madrid_Power_Plant_{ver}.nc'
    fpath  = fpath + suffix
    if not os.path.isfile(fpath):
        print(f"  [warn] trop file not found: {os.path.basename(fpath)}")
        return None, None

    with xr.open_dataset(fpath) as ds:
        ## Shape: (1, scanline, ground_pixel) and (1, scanline, ground_pixel, 4)
        lat    = ds['latitude'].values.squeeze()                          # (S, P)
        lon    = ds['longitude'].values.squeeze()
        no2    = ds['nitrogendioxide_tropospheric_column'].values.squeeze() * 1e6  # mol/m2 -> µmol/m2
        qa     = ds['qa_value'].values.squeeze()
        lat_b  = ds['latitude_bounds'].values.squeeze()                   # (S, P, 4)
        lon_b  = ds['longitude_bounds'].values.squeeze()

    mask = ((lat > lat_c - buf) & (lat < lat_c + buf) &
            (lon > lon_c - buf) & (lon < lon_c + buf) &
            (qa >= qa_min) & np.isfinite(no2) & (no2 > 0))

    if not mask.any():
        return None, None

    ## Build (N, 4, 2) polygon array [lon, lat]
    lon_b_flat = lon_b[mask]   # (N, 4)
    lat_b_flat = lat_b[mask]
    verts = np.stack([lon_b_flat, lat_b_flat], axis=-1)  # (N, 4, 2)
    return verts, no2[mask]


def _pace_polygons(fpath, lat_c, lon_c, buf):
    """
    Load PACE pixel polygons reconstructed from centre coordinates.
    Since PACE has no corner arrays, corners are derived as midpoints
    between adjacent pixel centres — giving exact gap-free tiling.
    Returns (verts, values) arrays or (None, None).
    verts : (N, 4, 2)  [lon, lat] per corner
    values: (N,)       µmol/m²
    """
    if not os.path.isfile(fpath):
        return None, None

    with xr.open_dataset(fpath) as ds:
        lat_2d = ds['latitude'].values               # (nlines, npix)
        lon_2d = ds['longitude'].values
        no2_2d = ds['total_column_no2'].values.astype(float)
        qf_2d  = ds['quality_no2'].values

    nL, nP = lat_2d.shape

    ## --- build corner grids via midpoints along each axis ---
    ## Along-track (line) midpoints: shape (nL+1, nP)
    lat_mid_L = np.empty((nL + 1, nP), dtype=np.float32)
    lon_mid_L = np.empty((nL + 1, nP), dtype=np.float32)
    lat_mid_L[1:-1] = 0.5 * (lat_2d[:-1] + lat_2d[1:])
    lon_mid_L[1:-1] = 0.5 * (lon_2d[:-1] + lon_2d[1:])
    ## Extrapolate edges
    lat_mid_L[0]  = lat_2d[0]  - (lat_mid_L[1]  - lat_2d[0])
    lat_mid_L[-1] = lat_2d[-1] + (lat_2d[-1]    - lat_mid_L[-2])
    lon_mid_L[0]  = lon_2d[0]  - (lon_mid_L[1]  - lon_2d[0])
    lon_mid_L[-1] = lon_2d[-1] + (lon_2d[-1]    - lon_mid_L[-2])

    ## Cross-track (pixel) midpoints: shape (nL+1, nP+1)
    lat_corners = np.empty((nL + 1, nP + 1), dtype=np.float32)
    lon_corners = np.empty((nL + 1, nP + 1), dtype=np.float32)
    lat_corners[:, 1:-1] = 0.5 * (lat_mid_L[:, :-1] + lat_mid_L[:, 1:])
    lon_corners[:, 1:-1] = 0.5 * (lon_mid_L[:, :-1] + lon_mid_L[:, 1:])
    ## Extrapolate cross-track edges
    lat_corners[:, 0]  = lat_mid_L[:, 0]  - (lat_corners[:, 1]  - lat_mid_L[:, 0])
    lat_corners[:, -1] = lat_mid_L[:, -1] + (lat_mid_L[:, -1]   - lat_corners[:, -2])
    lon_corners[:, 0]  = lon_mid_L[:, 0]  - (lon_corners[:, 1]  - lon_mid_L[:, 0])
    lon_corners[:, -1] = lon_mid_L[:, -1] + (lon_mid_L[:, -1]   - lon_corners[:, -2])
    ## Corner grid is (nL+1, nP+1); pixel [i,j] uses corners [i,j],[i,j+1],[i+1,j+1],[i+1,j]

    ## Flat mask
    mask = ((lat_2d > lat_c - buf) & (lat_2d < lat_c + buf) &
            (lon_2d > lon_c - buf) & (lon_2d < lon_c + buf) &
            np.isfinite(no2_2d) & (no2_2d > 0) & (qf_2d < 0.5))

    if not mask.any():
        return None, None

    ii, jj = np.where(mask)

    ## Extract 4 corners per pixel: TL, TR, BR, BL  (counter-clockwise in lon/lat)
    tl_lon = lon_corners[ii,   jj  ];  tl_lat = lat_corners[ii,   jj  ]
    tr_lon = lon_corners[ii,   jj+1];  tr_lat = lat_corners[ii,   jj+1]
    br_lon = lon_corners[ii+1, jj+1];  br_lat = lat_corners[ii+1, jj+1]
    bl_lon = lon_corners[ii+1, jj  ];  bl_lat = lat_corners[ii+1, jj  ]

    verts = np.stack([
        np.stack([tl_lon, tl_lat], axis=-1),
        np.stack([tr_lon, tr_lat], axis=-1),
        np.stack([br_lon, br_lat], axis=-1),
        np.stack([bl_lon, bl_lat], axis=-1),
    ], axis=1)   # (N, 4, 2)

    ## molecules/cm2 -> µmol/m2
    no2_umol = no2_2d[mask] / 6.022e23 * 1e4 * 1e6
    return verts, no2_umol


## --------------------------------------------------------------------------
## PLOT PANEL
## --------------------------------------------------------------------------

def _plot_panel(ax, verts, values, vmin, vmax, cmap, lat_c, lon_c, buf, title, t_str, dt_min):
    """Draw one map panel with filled pixel polygons."""
    ax.set_xlim(lon_c - buf, lon_c + buf)
    ax.set_ylim(lat_c - buf, lat_c + buf)
    ax.tick_params(left=False, right=False, bottom=False, top=False,
                   labelleft=False, labelbottom=False)

    ctx.add_basemap(ax, crs='EPSG:4326', alpha=0.6, attribution=False,
                    source=ctx.providers.OpenStreetMap.Mapnik)

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)
    col = PolyCollection(verts, array=values, cmap=cmap_obj, norm=norm,
                         linewidths=0, alpha=0.85, zorder=5)
    ax.add_collection(col)

    ax.scatter(lon_c, lat_c, marker='*', color='red', s=60, zorder=10)
    ax.set_title(title, fontsize=7, pad=3)
    ax.text(0.02, 0.97, t_str, transform=ax.transAxes,
            color='white', fontsize=6, va='top',
            bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.5))
    if dt_min is not None:
        sign = '+' if dt_min >= 0 else ''
        ax.text(0.02, 0.88, f'Δt = {sign}{dt_min:.0f} min', transform=ax.transAxes,
                color='yellow', fontsize=6, va='top',
                bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.5))
    return col


## --------------------------------------------------------------------------
## MAIN
## --------------------------------------------------------------------------

def main(cfg):
    mpl.rcParams['font.size'] = 7.

    df_sum  = pd.read_csv(cfg['f_summary'])
    df_trop = pd.read_csv(cfg['tropomi_csv'])
    df_trop['t_mid'] = df_trop.apply(
        lambda r: pd.Timestamp(r['time_utc_i'].replace('Z','')) +
                  (pd.Timestamp(r['time_utc_f'].replace('Z','')) -
                   pd.Timestamp(r['time_utc_i'].replace('Z',''))) / 2,
        axis=1)

    os.makedirs(cfg['d_out'], exist_ok=True)
    lat_c = cfg['target_lat']
    lon_c = cfg['target_lon']
    buf   = cfg['map_buffer']
    ver   = CT.PRCS_VER['trop']

    df_pairs = df_sum.loc[
        (df_sum['overlap'] == True) &
        (df_sum['n_pts']   >  0) &
        (df_sum['dt_min'].abs() <= cfg['dt_max_hr'] * 60)
    ].reset_index(drop=True)
    print(f"Co-located pairs to plot: {len(df_pairs)}")

    for i, row in df_pairs.iterrows():
        pace_file = row['pace_file']
        trop_file = row['trop_file']
        t_pace    = pd.Timestamp(row['t_pace'])
        dt_min    = float(row['dt_min'])
        t_trop_mid = pd.Timestamp(row['t_trop_mid']) if pd.notna(row.get('t_trop_mid','')) else None

        print(f"[{i+1}/{len(df_pairs)}] {pace_file}  dt={dt_min:+.1f} min")

        pace_path = os.path.join(cfg['d_pace_nc'], pace_file)
        trop_base = os.path.join(cfg['d_trop_nc'], trop_file)

        verts_pace, vals_pace = _pace_polygons(pace_path, lat_c, lon_c, buf)
        verts_trop, vals_trop = _trop_polygons(trop_base, ver, cfg['trop_qa_min'], lat_c, lon_c, buf)

        if verts_pace is None or verts_trop is None:
            print(f"  skip: pace={'ok' if verts_pace is not None else 'empty'}  "
                  f"trop={'ok' if verts_trop is not None else 'empty'}")
            continue

        ## Shared colorbar range across both datasets
        combined = np.concatenate([vals_pace, vals_trop])
        combined = combined[np.isfinite(combined)]
        vmin = float(np.percentile(combined, cfg['pct_lo'] * 100))
        vmax = float(np.percentile(combined, cfg['pct_hi'] * 100))

        t_trop_str = t_trop_mid.strftime('%Y-%m-%d %H:%M UTC') if t_trop_mid else ''
        t_pace_str = t_pace.strftime('%Y-%m-%d %H:%M UTC')

        fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.2),
                                 gridspec_kw={'wspace': 0.06})
        fig.subplots_adjust(right=0.90)

        _plot_panel(axes[0], verts_trop, vals_trop, vmin, vmax, cfg['cmap'],
                    lat_c, lon_c, buf,
                    title=r'TROPOMI NO$_2$ [$\mu$mol m$^{-2}$]',
                    t_str=t_trop_str, dt_min=None)
        _plot_panel(axes[1], verts_pace, vals_pace, vmin, vmax, cfg['cmap'],
                    lat_c, lon_c, buf,
                    title=r'PACE OCI NO$_2$ [$\mu$mol m$^{-2}$]',
                    t_str=t_pace_str, dt_min=dt_min)

        cbar_ax = fig.add_axes([0.92, 0.12, 0.018, 0.74])
        cb = fig.colorbar(plt.cm.ScalarMappable(
                 cmap=cfg['cmap'], norm=mcolors.Normalize(vmin=vmin, vmax=vmax)),
                 cax=cbar_ax)
        cb.set_label(r'NO$_2$ [$\mu$mol m$^{-2}$]', fontsize=7)
        cb.ax.tick_params(labelsize=6)

        date_str = t_pace.strftime('%Y-%m-%d')
        fig.suptitle(f'New Madrid Power Plant — {date_str}  |  Δt = {dt_min:+.0f} min',
                     fontsize=8, y=0.98)

        tag   = t_pace.strftime('%Y%m%dT%H%M%S')
        f_out = os.path.join(cfg['d_out'], f'pace_tropomi_2167_{tag}.png')
        fig.savefig(f_out, dpi=180, bbox_inches='tight')
        plt.close('all')
        print(f"  saved: {os.path.basename(f_out)}")

    print(f"\nDone. Figures in: {cfg['d_out']}")


if __name__ == '__main__':
    main(CFG)

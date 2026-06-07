import os
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib as mpl
mpl.use('Agg', force=True)
import matplotlib.pyplot as plt
import contextily as ctx
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation
import sys
sys.path.insert(0, os.pardir)
import CT, FN

# =============================================================================
# csf_plot.py
#
# Three panels per figure:
#   1. Map      — basemap + satellite pixels + trajectory + transects
#   2. Gaussian — per-transect distance vs. NO2 with Gaussian fit
#   3. Time     — flux / wind speed vs. transport time (two rows)
#
# Two figures produced per overpass:
#   <dfout>.png          primary, uses optimised trajectory (_O columns)
#   <dfout>_hysplit.png  comparison (CFG["o_plot_hysplit"]=True), uses _H columns
#
# Grid layout  9 rows × 12 cols:
#   rows 0-3, cols 0-2   map
#   rows 0-3, cols 4-11  Gaussian panels (4×8 = 32 slots)
#   col 3 / row 4        spacers
#   rows 5-6, cols 0-11  flux vs. time
#   rows 7-8, cols 0-11  wind speed vs. time
# =============================================================================


def _style(satellite):
    if satellite == "trop":
        return dict(voi="no2", flux_var="flux_no2",
                    clr_ortho="white", clr_text="white",
                    label=r"TROPOMI NO$_2$ [$\mu$mol m$^{-2}$]",
                    cmap="viridis", pct=(0.00, 0.99),
                    ylim_pad=lambda lo, hi: (0., hi * 1.10))
    return dict(voi="xco2", flux_var="flux_co2",
                clr_ortho="black", clr_text="black",
                label=r"OCO-3 XCO$_2$ [ppm]",
                cmap="viridis", pct=(0.01, 0.95),
                ylim_pad=lambda lo, hi: (lo - 0.05, hi + 0.05))


def _fmt_sigpct(val):
    return "> 100" if val > 100. else str(val)


def _scatter_qf(ax, df, age_col, val_col, color, suffix, marker="o", label=None):
    """Scatter split by QF flag: filled = good, x = bad."""
    qf = f"QF_gauss_abs{suffix}"
    if qf in df.columns:
        good = df[df[qf] == 0]
        bad  = df[df[qf] == 1]
        ax.scatter(good[age_col], good[val_col], marker=marker, color=color, label=label, zorder=5)
        ax.scatter(bad[age_col],  bad[val_col],  marker="x",    color=color, zorder=5)
    else:
        ax.scatter(df[age_col], df[val_col], marker=marker, color=color, label=label, zorder=5)


def _draw_plume_boundary(ax, plume_lons, plume_lats, xlim, ylim):
    """
    Draw the true outer boundary of the plume pixel set.
    Rasterise pixel coordinates onto a fine grid, dilate by 1 cell,
    then use contour at level 0.5 to trace the exact shape — handles
    curved and concave plumes correctly (convex hull would not).
    """
    if plume_lons is None or len(plume_lons) < 3:
        return
    nx, ny = 300, 300
    xi = np.linspace(xlim[0], xlim[1], nx)
    yi = np.linspace(ylim[0], ylim[1], ny)
    dx, dy = xi[1] - xi[0], yi[1] - yi[0]
    grid = np.zeros((ny, nx), dtype=float)
    ix = np.clip(((np.asarray(plume_lons) - xlim[0]) / dx).astype(int), 0, nx - 1)
    iy = np.clip(((np.asarray(plume_lats) - ylim[0]) / dy).astype(int), 0, ny - 1)
    grid[iy, ix] = 1.0
    grid = binary_dilation(grid).astype(float)
    ax.contour(xi, yi, grid, levels=[0.5], colors=["orange"], linewidths=[0.9], zorder=6)


# =============================================================================
# PANEL 1 — map
# =============================================================================

def _plot_map(ax, CFG, target, data, data_sampled, data_csf_s,
              traj, sate, st, o_plot_wddiff, suffix,
              plume_lons=None, plume_lats=None):
    """
    data_csf_s : CSF table with active suffix stripped to bare column names
    traj       : trajectory to display (true_tdump for _O, tdump for _H)
    suffix     : "_O" or "_H"
    """
    ax.set_xlim([target["lon"] - 1.65 * 1.1, target["lon"] + 1.65 * 1.1])
    ax.set_ylim([target["lat"] - 1.70 * 1.1, target["lat"] + 1.70 * 1.1])
    ax.tick_params(left=False, right=False, bottom=False, top=False,
                   labelleft=False, labelbottom=False)

    ctx.add_basemap(ax, alpha=1.0, crs="EPSG:4326", attribution=False,
                    source=ctx.providers.OpenStreetMap.Mapnik)
    ax.scatter(target["lon"], target["lat"], marker="*", color="red", s=30, zorder=10)

    # satellite pixels + colourbar
    voi = st["voi"]
    if CFG["SATE_INFO"][0] == "trop":
        data = data.loc[data["qa_value"] > 0.5]
    lo = data[voi].sort_values().iloc[int(len(data) * st["pct"][0])]
    hi = data[voi].sort_values().iloc[int(len(data) * st["pct"][1])]
    data.plot(ax=ax, column=voi, cmap=st["cmap"], alpha=1.0, vmin=lo, vmax=hi, legend=False)
    cax = ax.inset_axes([-0.05, 0.65, 0.03, 0.35])
    cb  = plt.colorbar(plt.cm.ScalarMappable(cmap=st["cmap"],
                       norm=plt.Normalize(vmin=lo, vmax=hi)), cax=cax, orientation="vertical")
    cb.ax.yaxis.set_ticks_position("left")
    cb.ax.yaxis.set_label_position("left")
    cb.ax.tick_params(labelsize=6)
    cb.ax.text(0.5, 1.05, st["label"], transform=cb.ax.transAxes, va="bottom", ha="left", fontsize=6)

    # plume boundary (primary figure only)
    _draw_plume_boundary(ax, plume_lons, plume_lats, ax.get_xlim(), ax.get_ylim())

    # trajectory
    ax.plot(traj["longitude"].values, traj["latitude"].values,
            color="white", lw=0.8, linestyle="solid", marker="o", markersize=0.1, zorder=7)

    # orthogonal transects + a4 FWHM (red) + a3 centre (red open star)
    for tid in data_sampled["tdump_id"].unique():
        if tid not in data_csf_s["tdump_id"].values:
            continue
        row  = data_csf_s.loc[data_csf_s["tdump_id"] == tid].iloc[0]
        seg  = data_sampled.loc[data_sampled["tdump_id"] == tid]
        ax.plot(seg["lon_ortho"], seg["lat_ortho"],
                linestyle="dashed", color=st["clr_ortho"], lw=0.35)
        fwhm = seg.loc[(seg["dist_ortho"] >= row["a3"] - row["a4"] / 2) &
                       (seg["dist_ortho"] <= row["a3"] + row["a4"] / 2)]
        ax.plot(fwhm["lon_ortho"], fwhm["lat_ortho"],
                linestyle="solid", color="red", lw=0.50, zorder=8)
        ax.scatter(row["lon_a3"], row["lat_a3"],
                   marker="*", edgecolor="red", facecolor="none", s=15, lw=0.6, zorder=9)

    if o_plot_wddiff and "wd_diff_avg" in data_csf_s.columns:
        for i in range(len(data_csf_s)):
            r = data_csf_s.iloc[i]
            ax.plot(r[["lon_a3_i", "lon_a3_f"]].values,
                    r[["lat_a3_i", "lat_a3_f"]].values, color="yellow", lw=0.5)
            if i % 2 == 0:
                ax.text(r["lon_a3"], r["lat_a3"],
                        str(int(r["wd_diff_avg"])), fontsize=3, color="yellow")

    lbl = "Optimised" if suffix == "_O" else "HYSPLIT"
    ax.text(0.04, 0.96,
            f"{pd.to_datetime(sate['time_utc']).strftime('%Y-%m-%d %H:%M')} UTC  [{lbl}]",
            va="center", transform=ax.transAxes, color=st["clr_text"], fontsize=6)


# =============================================================================
# PANEL 2 — per-transect Gaussian fit panels
# =============================================================================

def _plot_gaussian(ax, pp_list, CFG, data_csf_s, data_sampled, st):
    """
    data_csf_s  : CSF table with active suffix stripped (bare column names)
    data_sampled: pixels sampled along the active trajectory
    """
    voi, flux_var = st["voi"], st["flux_var"]
    xlim = [-110., 110.]
    valid = [t for t in data_csf_s["tdump_id"].values
             if t in data_sampled["tdump_id"].unique()]
    if not valid:
        return
    all_vals = pd.concat([data_sampled.loc[data_sampled["tdump_id"] == t, voi] for t in valid])
    ylim = st["ylim_pad"](all_vals.min(), all_vals.max())

    for i, pp in enumerate(pp_list[:len(data_csf_s)]):
        row     = data_csf_s.loc[i]
        is_last = (pp == pp_list[min(len(data_csf_s) - 1, len(pp_list) - 1)])
        ax[pp].axis("on")
        ax[pp].tick_params(axis="x", bottom=True, top=False, labelbottom=is_last,
                           direction="in", left=False, right=False, labelleft=False)
        ax[pp].tick_params(axis="y", right=True, left=False, labelright=is_last,
                           labelleft=False, direction="in")
        if is_last:
            ax[pp].set_xlabel("Distance [km]", fontsize=5)
        ax[pp].set_xlim(xlim)
        ax[pp].set_ylim(ylim)

        tmp = data_sampled.loc[data_sampled["tdump_id"] == row["tdump_id"]].reset_index(drop=True)
        ax[pp].scatter(tmp["dist_ortho"], tmp[voi],
                       marker="o", edgecolor="grey", facecolor="none", s=2., lw=0.4)

        params = row[["a0", "a1", "a2", "a3", "a4"]].values
        qf_ok  = "QF_gauss_abs" in data_csf_s.columns and row["QF_gauss_abs"] == 0
        ax[pp].plot(np.arange(xlim[0], xlim[1]),
                    FN._gaussian(np.arange(xlim[0], xlim[1]), *params),
                    color="limegreen" if qf_ok else "grey", lw=1.0)

        # a4 FWHM span (red) + a3 centre (red open star)
        x1, x2 = row["a3"] - row["a4"] / 2, row["a3"] + row["a4"] / 2
        y1, y2 = FN._gaussian(x1, *params), FN._gaussian(x2, *params)
        ax[pp].plot([x1, x2], [y1, y2], color="red", lw=0.70)
        ax[pp].scatter(row["a3"], (y1 + y2) / 2,
                       marker="*", edgecolor="red", facecolor="none", s=15, lw=0.6)

        if not np.isnan(row[flux_var]):
            for k, (xpos, txt) in enumerate([
                (0.80, f"{int(row['tdump_id']) + 1}"),
                (0.03, f"Flux={np.round(row[flux_var], 2)}"),
                (0.03, fr"r²={np.round(row['rsq_detrend'], 2)}"),
                (0.03, fr"gap={int(np.round(row['d_gap']))}%"),
                (0.03, fr"σa2={_fmt_sigpct(np.round(row['a2sig']/(row['a2']+1e-9)*100.))}%"),
                (0.03, fr"σa4={_fmt_sigpct(np.round(row['a4sig']/(row['a4']+1e-9)*100.))}%"),
            ]):
                ax[pp].text(xpos, 1.0 - 0.14 * (k + 1), txt,
                            transform=ax[pp].transAxes, fontsize=3.5, color="black", ha="left")


# =============================================================================
# PANEL 3 — time series (flux and wind)
# =============================================================================

def _plot_timeseries(ax1, ax2, CFG, data_csf, nox_result, cems_nox, flux_unit, st, suffix):
    """
    ax1 : flux vs. transport time (upper)
    ax2 : wind speed vs. transport time (lower)
    Both show only the active suffix (_O or _H) to match the map.
    PSS ratio is on a secondary y-axis of ax1.
    Both axes share a secondary x-axis showing transport distance [km].
    """
    age_col  = f"age_hours{suffix}"
    flux_col = f"{st['flux_var']}{suffix}"
    wso_col  = f"wso{suffix}"
    pss_col  = f"pss_ratio{suffix}"
    fit_col  = f"flux_nox_fit{suffix}"
    km_col   = f"age_km{suffix}" if f"age_km{suffix}" in data_csf.columns else "age_km"
    clr      = "limegreen" if suffix == "_O" else "dodgerblue"
    lbl      = "Optimised" if suffix == "_O" else "HYSPLIT"

    # build hrs↔kms lookup once for both secondary x-axes
    hrs, kms = np.array([]), np.array([])
    if km_col in data_csf.columns:
        _df  = data_csf[[age_col, km_col]].dropna().drop_duplicates(subset=age_col).sort_values(age_col)
        hrs  = _df[age_col].values
        kms  = _df[km_col].values

    def _add_km_axis(ax):
        if len(hrs) >= 2:
            ax2x = ax.secondary_xaxis(
                "top",
                functions=(lambda h: np.interp(h, hrs, kms),
                           lambda k: np.interp(k, kms, hrs)))
            ax2x.set_xlabel("Transport distance [km]", fontsize=6)
            ax2x.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))

    # ---- upper panel: flux ----
    ax1.set_ylabel(f"Flux {flux_unit}")
    _scatter_qf(ax1, data_csf, age_col, flux_col, clr, suffix=suffix, label=f"CSF ({lbl})")
    _add_km_axis(ax1)

    # PSS ratio on secondary y-axis
    if pss_col in data_csf.columns and data_csf[pss_col].notna().any():
        ax_pss = ax1.twinx()
        valid  = data_csf[[age_col, pss_col]].dropna()
        ax_pss.scatter(valid[age_col], valid[pss_col],
                       marker="^", color="purple", s=12, label="NOx/NO2 (PSS)", zorder=4)
        ax_pss.set_ylabel("NOx/NO2 PSS ratio", color="purple", fontsize=7)
        ax_pss.tick_params(axis="y", colors="purple")
        ax_pss.legend(fontsize=6, loc="upper left")

    # fitted NOx decay curve
    if fit_col in data_csf.columns:
        fit = data_csf[[age_col, fit_col]].dropna().sort_values(age_col)
        if not fit.empty:
            ax1.plot(fit[age_col], fit[fit_col],
                     color="orangered", lw=1.2, linestyle="--", label="NOx fit")

    # NOx workflow scalar annotations (primary figure only)
    if nox_result and nox_result.get("status") == "ok":
        age_pss = nox_result["age_at_PSS_s"] / 3600.
        ax1.axvline(age_pss, color="purple", linestyle="dotted", lw=1.0, label="PSS point")
        ax1.scatter(age_pss, nox_result["flux_nox_PSS"],
                    marker="*", color="orangered", s=80, zorder=6, label="NOx@PSS")
        if not np.isnan(nox_result.get("flux_nox_source", np.nan)):
            ax1.scatter(0, nox_result["flux_nox_source"],
                        marker="D", color="orangered", s=40, zorder=6, label="NOx source")
        opt   = nox_result.get("option", "?")
        k_val = nox_result.get("k_loss") if opt == "B" else nox_result.get("k_eff")
        k_lbl = "k_loss" if opt == "B" else "k_eff"
        lines = [f"NOx workflow (opt {opt})",
                 f"PSS= {nox_result['pss_ratio']:.3f}",
                 f"NOx@PSS= {nox_result['flux_nox_PSS']:.3f} {flux_unit}"]
        if opt == "B" and not np.isnan(nox_result.get("oh_nd", np.nan)):
            lines.append(f"[OH]= {nox_result['oh_nd']:.2e} /cm³")
        if k_val and not np.isnan(k_val):
            lines.append(f"{k_lbl}= {k_val:.2e} s⁻¹")
        if not np.isnan(nox_result.get("flux_nox_source", np.nan)):
            lines.append(f"NOx src= {nox_result['flux_nox_source']:.3f} {flux_unit}")
        ax1.text(0.02, 0.97, "\n".join(lines), transform=ax1.transAxes, fontsize=5.5,
                 va="top", color="orangered",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="orangered", alpha=0.8))

    # CEMS overlay
    if cems_nox is not None and not cems_nox.empty:
        cems = cems_nox.copy()
        cems["transport_hrs"] = (cems["tstmp_utc"].max() - cems["tstmp_utc"]).dt.total_seconds() / 3600.
        ax1.step(cems["transport_hrs"], cems["noxMass_tph_metric"],
                 where="mid", color="darkorange", lw=1.2, label="CEMS NOx")
        ax1.scatter(cems["transport_hrs"], cems["noxMass_tph_metric"],
                    marker="s", color="darkorange", s=18, zorder=5)
        ax1.text(0.02, 0.04,
                 f"CEMS= {cems['noxMass_tph_metric'].mean():.2f} {flux_unit}",
                 transform=ax1.transAxes, fontsize=6, color="darkorange")

    ax1.legend(fontsize=6, loc="upper right")

    # ---- lower panel: wind speed ----
    ax2.set_xlabel("Transport time [hours]")
    ax2.set_ylabel("Wind speed [m/s]")
    if wso_col in data_csf.columns:
        _scatter_qf(ax2, data_csf, age_col, wso_col, clr, suffix=suffix, label=f"wso ({lbl})")
    _add_km_axis(ax2)
    ax2.legend(fontsize=6, loc="upper right")


# =============================================================================
# FIGURE BUILDER + PUBLIC ENTRY POINT
# =============================================================================

def _build_figure(CFG, target, data, data_sampled, trop_csf, traj,
                  sate, flux_unit, o_plot_wddiff, dfout,
                  suffix, cems_nox, nox_result, plume_lons, plume_lats):
    """Build and save one CSF figure for the given suffix."""
    mpl.rcParams['font.size'] = 6.
    fig, ax = plt.subplots(
        nrows=9, ncols=12, figsize=(7.24, 5.00),
        gridspec_kw=dict(width_ratios=[1,1,1,0.1,1,1,1,1,1,1,1,1],
                         height_ratios=[1,1,1,1,0.05,1,1,1,1]))
    plt.subplots_adjust(left=0.07, bottom=0.07, right=0.930, top=0.93, wspace=0, hspace=0)
    for j in range(9):
        for i in range(12):
            ax[j, i].axis("off")

    gs      = ax[0, 0].get_gridspec()
    ax_map  = fig.add_subplot(gs[0:4, 0:3])
    ax_ts1  = fig.add_subplot(gs[5:7, 0:12])
    ax_ts2  = fig.add_subplot(gs[7:9, 0:12])
    pp_list = [(r, c) for r in range(4) for c in range(4, 12)]
    st      = _style(CFG["SATE_INFO"][0])

    # strip active suffix for map/Gaussian panels (bare column names)
    csf_s = trop_csf.rename(columns=lambda c: c[:-len(suffix)] if c.endswith(suffix) else c)

    _plot_map(ax_map, CFG, target, data, data_sampled, csf_s,
              traj, sate, st, o_plot_wddiff, suffix, plume_lons, plume_lats)
    _plot_gaussian(ax, pp_list, CFG, csf_s, data_sampled, st)
    _plot_timeseries(ax_ts1, ax_ts2, CFG, trop_csf, nox_result,
                     cems_nox, flux_unit, st, suffix)

    fig.savefig(dfout + ".png", dpi=200)
    plt.close("all")


def _plot_csf(CFG, target, data, data_sampled, data_csf, tdump,
              sate, flux_unit, o_plot_wddiff, dfout,
              cems_nox=None, true_tdump=None, plume_lons=None, plume_lats=None,
              nox_result=None, data_true_sampled=None):
    """
    Primary figure  (_O): true_tdump, optimised transects, plume boundary.
    Comparison figure (_H, optional): tdump, HYSPLIT transects, no plume boundary.
    """
    if data_true_sampled is None:
        data_true_sampled = pd.DataFrame()

    _build_figure(CFG, target, data,
                  data_sampled  = data_true_sampled,
                  trop_csf      = data_csf,
                  traj          = true_tdump if true_tdump is not None else tdump,
                  sate=sate, flux_unit=flux_unit, o_plot_wddiff=o_plot_wddiff,
                  dfout=dfout, suffix="_O", cems_nox=cems_nox,
                  nox_result=nox_result, plume_lons=plume_lons, plume_lats=plume_lats)

    if CFG.get("o_plot_hysplit", False):
        _build_figure(CFG, target, data,
                      data_sampled  = data_sampled,
                      trop_csf      = data_csf,
                      traj          = tdump,
                      sate=sate, flux_unit=flux_unit, o_plot_wddiff=o_plot_wddiff,
                      dfout=dfout + "_hysplit", suffix="_H", cems_nox=cems_nox,
                      nox_result=None, plume_lons=None, plume_lats=None)

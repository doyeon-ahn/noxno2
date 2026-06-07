import os
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib as mpl
mpl.use('Agg', force=True)
import matplotlib.pyplot as plt
import contextily as ctx
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
sys.path.insert(0, os.pardir)
import CT, FN

# =============================================================================
# csf_plot.py
# Figure layout  (9 rows × 12 cols):
#   rows 0–3, cols 0–2   : map panel
#   rows 0–3, cols 4–11  : per-transect Gaussian panels (4×8 = 32 slots)
#   col 3                : narrow spacer between map and Gaussian panels
#   row 4                : narrow spacer between map/Gaussian and time series
#   rows 5–6, cols 0–11  : flux vs. transport time  (full width)
#   rows 7–8, cols 0–11  : wind speed vs. transport time  (full width)
#
# The time series panels span the full 12-column width, so their left edge
# aligns with the map and their right edge aligns with the Gaussian panels.
#
# Produces one figure per overpass (<dfout>.png) using the optimised
# trajectory (_O columns).  When CFG["o_plot_hysplit"]=True, also saves
# <dfout>_hysplit.png using the original HYSPLIT trajectory (_H columns)
# for side-by-side comparison.
# =============================================================================


# =============================================================================
# HELPERS
# =============================================================================

def _satellite_style(satellite):
    if satellite == "trop":
        return dict(voi="no2", flux_var="flux_no2",
                    clr_ortho="white", clr_text="white",
                    label=r"TROPOMI NO$\mathregular{_{2}}$ [$\mu$mol m$\mathregular{^{-2}}$]",
                    label_short=r"NO$\mathregular{_{2}}$ [$\mu$mol m$\mathregular{^{-2}}$]",
                    cmap="viridis", pct=(0.00, 0.99),
                    ylim_pad=lambda lo, hi: (0., hi * 1.10))
    if satellite == "oco3":
        return dict(voi="xco2", flux_var="flux_co2",
                    clr_ortho="black", clr_text="black",
                    label=r"OCO-3 SAM XCO$\mathregular{_{2}}$ [ppm]",
                    label_short=r"XCO$\mathregular{_{2}}$ [ppm]",
                    cmap="viridis", pct=(0.01, 0.95),
                    ylim_pad=lambda lo, hi: (lo - 0.05, hi + 0.05))


def _fmt_sigpct(val):
    return "> 100" if val > 100. else str(val)


def _scatter_qf(ax, data_csf, age_col, val_col, color, suffix="_H", marker="o", label=None):
    """Scatter plot split by QF_gauss_abs{suffix}: filled=good, 'x'=bad."""
    qf = "QF_gauss_abs" + suffix
    if qf in data_csf.columns:
        good = data_csf[data_csf[qf] == 0]
        bad  = data_csf[data_csf[qf] == 1]
        ax.scatter(good[age_col], good[val_col], marker=marker, color=color, label=label, zorder=5)
        ax.scatter(bad[age_col],  bad[val_col],  marker="x",    color=color, zorder=5)
    else:
        ax.scatter(data_csf[age_col], data_csf[val_col], marker=marker,
                   edgecolor=color, facecolor="none", label=label, zorder=5)


def _convex_hull_boundary(plume_lons, plume_lats):
    """
    Compute the convex hull of the flood-fill plume pixel set and return
    closed boundary arrays for plotting as an outline polygon.
    Returns (None, None) if fewer than 3 points or computation fails.
    """
    if plume_lons is None or len(plume_lons) < 3:
        return None, None
    try:
        pts  = np.column_stack([plume_lons, plume_lats])
        hull = ConvexHull(pts)
        idx  = np.append(hull.vertices, hull.vertices[0])   # close the polygon
        return pts[idx, 0], pts[idx, 1]
    except Exception:
        return None, None


# =============================================================================
# PANEL 1 — basemap + satellite pixels + trajectory + transects
# =============================================================================

def _plot_map(ax, CFG, target, data, data_sampled, data_true_sampled,
              data_csf_H, tdump, sate, style, o_plot_wddiff,
              true_tdump=None, plume_lons=None, plume_lats=None):
    """
    Map panel.

    Shows the optimised trajectory (true_tdump, red) and the original HYSPLIT
    trajectory (tdump, white).  Transect orthogonal lines and a4 FWHM widths
    are drawn for both the HYSPLIT (_H, black) and optimised (_O, dodgerblue)
    fits where available.  The plume mask is drawn as a single orange convex-
    hull boundary line instead of individual pixel markers.
    """
    ax.set_xlim([target["lon"] - 1.65 * 1.1, target["lon"] + 1.65 * 1.1])
    ax.set_ylim([target["lat"] - 1.70 * 1.1, target["lat"] + 1.70 * 1.1])
    for axis in ["x", "y"]:
        ax.tick_params(axis=axis, which="both",
                       left=False, right=False, bottom=False, top=False,
                       labelleft=False, labelbottom=False)

    ctx.add_basemap(ax, alpha=1.0, crs="EPSG:4326", attribution=False,
                    source=ctx.providers.OpenStreetMap.Mapnik)
    ax.scatter(target["lon"], target["lat"], marker="*", color="red", s=30, zorder=10)

    # --- Satellite pixel colourmap ---
    voi = style["voi"]
    if CFG["SATE_INFO"][0] == "trop":
        data = data.loc[data["qa_value"] > 0.5]
    lo = data[voi].sort_values().iloc[int(len(data) * style["pct"][0])]
    hi = data[voi].sort_values().iloc[int(len(data) * style["pct"][1])]
    data.plot(ax=ax, column=voi, cmap=style["cmap"], alpha=1.0,
              vmin=lo, vmax=hi, legend=False)
    cax = ax.inset_axes([-0.05, 0.65, 0.03, 0.35])
    sm  = plt.cm.ScalarMappable(cmap=style["cmap"], norm=plt.Normalize(vmin=lo, vmax=hi))
    cb  = plt.colorbar(sm, cax=cax, orientation="vertical")
    cb.ax.yaxis.set_ticks_position("left")
    cb.ax.yaxis.set_label_position("left")
    cb.ax.tick_params(labelsize=6)
    cb.ax.text(0.5, 1.05, style["label"],
               transform=cb.ax.transAxes, va="bottom", ha="left")

    # --- Plume boundary: convex hull outline instead of per-pixel markers ---
    hull_lons, hull_lats = _convex_hull_boundary(plume_lons, plume_lats)
    if hull_lons is not None:
        ax.plot(hull_lons, hull_lats,
                color="orange", lw=0.9, linestyle="-", zorder=6)

    # --- Trajectories ---
    # Original HYSPLIT (white, thin) kept for spatial reference
    ax.plot(tdump["longitude"].values, tdump["latitude"].values,
            color="white", lw=0.6, linestyle="solid", marker="o", markersize=0.1)
    # Optimised trajectory (red) — primary
    if true_tdump is not None and len(true_tdump) > 1:
        ax.plot(true_tdump["longitude"].values, true_tdump["latitude"].values,
                color="red", lw=0.9, linestyle="solid", marker="o", markersize=0.0,
                zorder=7)

    # --- Transect segments ---
    # data_csf_H has _H stripped so bare column names (a3, a4, lon_a3, etc.)
    # _O columns keep their suffix (a3_O, a4_O) since only _H was stripped
    for tid in data_sampled["tdump_id"].unique():
        if tid not in data_csf_H["tdump_id"].values:
            continue
        row  = data_csf_H.loc[data_csf_H["tdump_id"] == tid].iloc[0]
        seg  = data_sampled.loc[data_sampled["tdump_id"] == tid]

        # Full orthogonal transect (dashed white)
        ax.plot(seg["lon_ortho"], seg["lat_ortho"],
                linestyle="dashed", color=style["clr_ortho"], lw=0.40)

        # HYSPLIT a4 FWHM width (black solid)
        fwhm = seg.loc[(seg["dist_ortho"] >= row["a3"] - row["a4"] / 2) &
                       (seg["dist_ortho"] <= row["a3"] + row["a4"] / 2)]
        ax.plot(fwhm["lon_ortho"], fwhm["lat_ortho"],
                linestyle="solid", color="black", lw=0.40)
        ax.scatter(row["lon_a3"], row["lat_a3"],
                   edgecolor="white", marker="o", s=3.0, lw=0.7, facecolor="none")

        # Optimised a4 FWHM width (dodgerblue solid) — only if _O columns present
        if (len(data_true_sampled) > 1
                and "a3_O" in data_csf_H.columns
                and "a4_O" in data_csf_H.columns):
            seg_true = data_true_sampled.loc[data_true_sampled["tdump_id"] == tid]
            fwhm_t   = seg_true.loc[
                (seg_true["dist_ortho"] >= row["a3_O"] - row["a4_O"] / 2) &
                (seg_true["dist_ortho"] <= row["a3_O"] + row["a4_O"] / 2)]
            ax.plot(fwhm_t["lon_ortho"], fwhm_t["lat_ortho"],
                    linestyle="solid", color="dodgerblue", lw=0.60, zorder=8)

    # --- Optional wind-direction comparison overlay ---
    if o_plot_wddiff:
        ax.plot(data_csf_H.iloc[:-1]["lon_a3"], data_csf_H.iloc[:-1]["lat_a3"],
                color="white", lw=0.7)
        ax.scatter(data_csf_H.iloc[:-1]["lon_a3"], data_csf_H.iloc[:-1]["lat_a3"],
                   edgecolor="white", marker="o", s=3.0, lw=0.7, facecolor="none")
        for i in range(len(data_csf_H)):
            ax.plot(data_csf_H.loc[i, ["lon_a3_i", "lon_a3_f"]].values,
                    data_csf_H.loc[i, ["lat_a3_i", "lat_a3_f"]].values,
                    color="black", lw=0.5)
            if i % 2 == 0:
                ax.text(data_csf_H.loc[i, "lon_a3"], data_csf_H.loc[i, "lat_a3"],
                        str(data_csf_H.loc[i, "tdump_id"]), fontsize=4)
                ax.text(data_csf_H.loc[i, "lon_a3"], data_csf_H.loc[i, "lat_a3"],
                        str(int(data_csf_H.loc[i, "wd_diff_avg"])), fontsize=4)

    ax.text(0.04, 0.96,
            f"{pd.to_datetime(sate['time_utc']).strftime('%Y-%m-%d %H:%M')} (UTC)",
            va="center", transform=ax.transAxes,
            color=style["clr_text"], fontsize=6)


# =============================================================================
# PANEL 2 — per-transect Gaussian fits
# =============================================================================

def _plot_gaussian(ax, pp_list, CFG, data_csf, data_sampled, data_true_sampled, style):
    """
    Per-transect distance vs. NO2 panels.

    Pixels sampled along the optimised trajectory (_O) are shown with the
    Gaussian fit.  Pixels sampled along HYSPLIT (_H) are available but
    commented out — uncomment the grey fit line if needed for comparison.
    """
    voi      = style["voi"]
    flux_var = style["flux_var"]
    xlim     = [-110., 110.]

    all_vals = pd.concat([data_sampled.loc[data_sampled["tdump_id"] == tid, voi]
                          for tid in data_csf["tdump_id"]])
    ylim = style["ylim_pad"](all_vals.min(), all_vals.max())

    for i, pp in enumerate(pp_list[:len(data_csf)]):
        row       = data_csf.loc[i]
        is_corner = (pp == (0, 11))
        ax[pp].axis("on")

        ax[pp].tick_params(axis="x", which="both", bottom=False, top=is_corner,
                           labelbottom=False, direction="in",
                           left=False, right=False, labeltop=is_corner)
        ax[pp].tick_params(axis="y", which="both", right=True, left=False,
                           labelright=is_corner, labelleft=False, direction="in",
                           bottom=False, top=False, labelbottom=False)
        if is_corner:
            ax[pp].set_xlabel("Distance [km]")
            ax[pp].xaxis.set_label_position("top")
            ax[pp].set_ylabel(style["label_short"])
            ax[pp].yaxis.set_label_position("right")
        ax[pp].set_xlim(xlim)
        ax[pp].set_ylim(ylim)

        # --- HYSPLIT-sampled pixels (grey Gaussian, commented out by default) ---
        if len(data_sampled) > 1:
            sfx    = "_H"
            params = row[["a0"+sfx, "a1"+sfx, "a2"+sfx, "a3"+sfx, "a4"+sfx]].values
            x_fit  = np.arange(xlim[0], xlim[1])
            # ax[pp].plot(x_fit, FN._gaussian(x_fit, *params),
            #             color='darkgrey', lw=1.0, ls="solid", zorder=1)

        # --- Optimised-sampled pixels + Gaussian fit (red) ---
        if len(data_true_sampled) > 1:
            sfx = "_O"
            tmp = data_true_sampled.loc[
                data_true_sampled["tdump_id"] == row["tdump_id"]
            ].reset_index(drop=True)
            ax[pp].scatter(tmp["dist_ortho"], tmp[voi],
                           marker="o", edgecolor="black", facecolor="none",
                           s=2., lw=0.4, alpha=0.6, zorder=2)

            qf_pass = ("QF_gauss_abs"+sfx in data_csf.columns
                       and row["QF_gauss_abs"+sfx] == 0)
            ls     = "solid" if qf_pass else "dotted"
            params = row[["a0"+sfx, "a1"+sfx, "a2"+sfx, "a3"+sfx, "a4"+sfx]].values
            x_fit  = np.arange(xlim[0], xlim[1])
            ax[pp].plot(x_fit, FN._gaussian(x_fit, *params),
                        color="red", lw=1.0, ls=ls, zorder=3)

            if not np.isnan(row[flux_var+sfx]):
                for k, (xpos, txt) in enumerate([
                    (0.85, f"{int(row['tdump_id']) + 1}"),
                    (0.03, f"Flux= {np.round(row[flux_var+sfx], 2)}"),
                    (0.03, fr"$\sigma_{{ctr}}$= {int(np.round(row['d_center'+sfx]))}"),
                    (0.03, fr"$\sigma_{{L}}$= {int(np.round(row['d_left'+sfx]))}"),
                    (0.03, fr"$\sigma_{{R}}$= {int(np.round(row['d_right'+sfx]))}"),
                    (0.03, fr"$\sigma_{{gap}}$= {int(np.round(row['d_gap'+sfx]))}"),
                    (0.03, fr"r$^2$= {np.round(row['rsq_detrend'+sfx], 2)}"),
                    (0.03, fr"$\sigma_{{a2}}$= {_fmt_sigpct(np.round(row['a2sig'+sfx] / (row['a2'+sfx] + 1e-9) * 100.))}"),
                    (0.03, fr"$\sigma_{{a4}}$= {_fmt_sigpct(np.round(row['a4sig'+sfx] / (row['a4'+sfx] + 1e-9) * 100.))}"),
                    (0.03, f"MAPE= {np.round(row['mape'+sfx], 4)}"),
                ]):
                    ax[pp].text(xpos, 1.06 - 0.105 * (k + 1), txt,
                                transform=ax[pp].transAxes,
                                fontsize=4.0, color="black", ha="left")


# =============================================================================
# PANEL 3 — flux vs. transport time  /  wind speed vs. transport time
# =============================================================================

def _plot_flux_vs_time(ax, CFG, data_csf, nox_result, cems_nox,
                       flux_unit, style, var_to_plot, nox_suffix="_O"):
    """
    Bottom panels: flux (var_to_plot='nox') or wind speed (var_to_plot='etc')
    vs. transport time.

    nox_suffix : suffix used in run_nox_workflow ("_O" or "_H"); determines
                 which per-row PSS and NOx fit columns to read from data_csf.
    """
    if var_to_plot == "nox":

        flux_var = style["flux_var"]
        age_H    = "age_hours_H"
        age_O    = "age_hours_O"

        ax.set_ylabel(f"Flux {flux_unit}")

        # --- CSF fluxes: HYSPLIT (blue) + Optimised (green) ---
        _scatter_qf(ax, data_csf, age_H, flux_var+"_H", "dodgerblue",
                    suffix="_H", label="CSF (HYSPLIT)")
        if flux_var+"_O" in data_csf.columns:
            _scatter_qf(ax, data_csf, age_O, flux_var+"_O", "limegreen",
                        suffix="_O", label="CSF (Optimised)")

        # --- Secondary x-axis: transport distance [km] ---
        age_km_col = "age_km_H" if "age_km_H" in data_csf.columns else "age_km"
        if age_km_col in data_csf.columns:
            _df = (data_csf[[age_H, age_km_col]].dropna()
                   .drop_duplicates(subset=age_H).sort_values(age_H))
            hrs, kms = _df[age_H].values, _df[age_km_col].values
            if len(hrs) >= 2:
                ax_km = ax.secondary_xaxis(
                    "top",
                    functions=(lambda h: np.interp(h, hrs, kms),
                               lambda k: np.interp(k, kms, hrs)))
                ax_km.set_xlabel("Transport distance [km]")
                ax_km.xaxis.set_major_formatter(
                    plt.FuncFormatter(lambda x, _: f"{x:.0f}"))

        # --- PSS ratio twin y-axis (Step 1, per-row, suffixed column) ---
        pss_col = f"pss_ratio{nox_suffix}"
        if pss_col in data_csf.columns and data_csf[pss_col].notna().any():
            ax_pss = ax.twinx()
            valid  = data_csf[[f"age_hours{nox_suffix}", pss_col]].dropna()
            ax_pss.scatter(valid[f"age_hours{nox_suffix}"], valid[pss_col],
                           marker="^", color="purple", s=12,
                           label="NOx/NO2 (PSS)", zorder=4)
            ax_pss.set_ylabel("NOx/NO2 PSS ratio", color="purple")
            ax_pss.tick_params(axis="y", colors="purple")
            ax_pss.legend(fontsize=6, loc="upper left")

        # --- Fitted NOx decay curve (Step 4, per-row, suffixed column) ---
        fit_col = f"flux_nox_fit{nox_suffix}"
        age_nox = f"age_hours{nox_suffix}"
        if fit_col in data_csf.columns:
            fit = data_csf[[age_nox, fit_col]].dropna().sort_values(age_nox)
            if not fit.empty:
                ax.plot(fit[age_nox], fit[fit_col],
                        color="orangered", lw=1.2, linestyle="--",
                        label="NOx fit (Step 4)")

        # --- NOx workflow scalar annotations ---
        if nox_result and nox_result.get("status") == "ok":
            age_pss_hr = nox_result["age_at_PSS_s"] / 3600.
            ax.axvline(age_pss_hr, color="purple", linestyle="dotted",
                       lw=1.0, label="PSS point")
            ax.scatter(age_pss_hr, nox_result["flux_nox_PSS"],
                       marker="*", color="orangered", s=80, zorder=6,
                       label="NOx flux @ PSS")
            ax.scatter(0, nox_result["flux_nox_source"],
                       marker="D", color="orangered", s=40, zorder=6,
                       label="NOx source")

            opt   = nox_result.get("option", "?")
            k_val = (nox_result.get("k_loss") if opt == "B"
                     else nox_result.get("k_eff"))
            k_lbl = "k_loss" if opt == "B" else "k_eff"
            oh_txt = (f"[OH]= {nox_result['oh_nd']:.2e} molec/cm³\n"
                      if opt == "B" else "")
            k_txt  = (f"{k_lbl}= {k_val:.2e} s⁻¹\n"
                      if k_val and not np.isnan(k_val) else "")
            info   = (f"── NOx workflow (opt {opt}) ──\n"
                      f"PSS ratio= {nox_result['pss_ratio']:.3f}  (Step 2)\n"
                      f"NOx@PSS=  {nox_result['flux_nox_PSS']:.3f} {flux_unit}  (Step 3)\n"
                      f"{oh_txt}{k_txt}"
                      f"NOx source= {nox_result['flux_nox_source']:.3f} {flux_unit}  (Step 4)")
            ax.text(0.02, 0.97, info, transform=ax.transAxes, fontsize=5.5,
                    va="top", color="orangered",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec="orangered", alpha=0.8))

        # --- Legacy NOx lifetime fit ---
        if CFG["SATE_INFO"][0] == "trop" and CFG.get("o_calc_nox"):
            _scatter_qf(ax, data_csf, age_H, "flux_nox_H", "orangered",
                        suffix="_H", label="NOx flux (HYSPLIT)")
            x = np.arange(0., 6., 0.1)
            ax.plot(x, FN._nox_lifetime(x, data_csf["emis_nox_H"].iloc[0],
                                         data_csf["t_pss_H"].iloc[0]),
                    color="orangered")
            ax.text(0.5, 0.85,
                    f"NOx Emission= {data_csf.loc[0,'emis_nox_H']:.1f} {flux_unit}",
                    transform=ax.transAxes, fontsize=7)
            ax.text(0.5, 0.75,
                    f"NOx Lifetime= {data_csf.loc[0,'nox_lifetime_H']:.1f} hrs",
                    transform=ax.transAxes, fontsize=7)
            if data_csf.loc[0, "t_pss_H"] != -9999.:
                ax.axvline(x=data_csf.loc[0, "t_pss_H"], color="grey",
                           linestyle="dotted")

        # --- CEMS overlay ---
        if cems_nox is not None and not cems_nox.empty:
            cems = cems_nox.copy()
            cems["transport_hrs"] = ((cems["tstmp_utc"].max() - cems["tstmp_utc"])
                                     .dt.total_seconds() / 3600.)
            ax.step(cems["transport_hrs"], cems["noxMass_tph_metric"],
                    where="mid", color="darkorange", lw=1.2, label="CEMS NOx")
            ax.scatter(cems["transport_hrs"], cems["noxMass_tph_metric"],
                       marker="s", color="darkorange", s=18, zorder=5)
            ax.text(0.02, 0.92,
                    f"CEMS mean= {cems['noxMass_tph_metric'].mean():.2f} {flux_unit}",
                    transform=ax.transAxes, fontsize=7, color="darkorange")

    if var_to_plot == "etc":
        ax.set_xlabel("Transport time [hours]")
        ax.set_ylabel("Wind speed [m/s]")
        _scatter_qf(ax, data_csf, "age_hours_H", "wso_H", "dodgerblue",
                    suffix="_H", label="HYSPLIT")
        if "age_hours_O" in data_csf.columns:
            _scatter_qf(ax, data_csf, "age_hours_O", "wso_O", "limegreen",
                        suffix="_O", label="Optimised")

    ax.legend(fontsize=6, loc="upper right")


# =============================================================================
# MAIN PLOT FUNCTION
# =============================================================================

def _plot_csf(CFG, target, data, data_sampled, data_csf, tdump,
              sate, flux_unit, o_plot_wddiff, dfout,
              cems_nox=None, true_tdump=None, plume_lons=None, plume_lats=None,
              nox_result=None, data_true_sampled=None):
    """
    Compose and save the three-panel CSF figure.

    Always saves <dfout>.png.
    If CFG["o_plot_hysplit"]=True, also saves <dfout>_hysplit.png with the
    map showing only the HYSPLIT trajectory and all panels using _H columns,
    intended for side-by-side comparison in presentations.
    """
    if data_true_sampled is None:
        data_true_sampled = pd.DataFrame()

    def _build(dfout_path, suffix_map, suffix_ts, traj, sampled, plume_l, plume_t):
        """
        Internal builder.  suffix_map/_ts separate because map always shows
        both trajectories while time series panels target one suffix.
        """
        mpl.rcParams['font.size'] = 6.
        fig, ax = plt.subplots(
            nrows=9, ncols=12, figsize=(7.24, 5.00),
            gridspec_kw=dict(
                width_ratios=[1, 1, 1, 0.1, 1, 1, 1, 1, 1, 1, 1, 1],
                height_ratios=[1, 1, 1, 1, 0.05, 1, 1, 1, 1]))
        plt.subplots_adjust(left=0.07, bottom=0.07, right=0.930,
                            top=0.93, wspace=0, hspace=0)
        for j in range(9):
            for i in range(12):
                ax[j, i].axis("off")

        gs       = ax[0, 0].get_gridspec()
        ax_map   = fig.add_subplot(gs[0:4,  0:3])
        ax_flux1 = fig.add_subplot(gs[5:7,  0:12])   # full width: cols 0–11
        ax_flux2 = fig.add_subplot(gs[7:9,  0:12])   # full width: cols 0–11
        pp_list  = [(r, c) for r in range(4) for c in range(4, 12)]  # 32 Gaussian slots
        style    = _satellite_style(CFG["SATE_INFO"][0])

        # Strip the map suffix so bare column names (a3, a4, ...) are used
        # in the map and Gaussian panels; _O columns keep their suffix.
        data_csf_stripped = data_csf.rename(
            columns=lambda c: c[:-2] if c.endswith(suffix_map) else c)

        _plot_map(ax_map, CFG, target, data, sampled, data_true_sampled,
                  data_csf_stripped, traj, sate, style, o_plot_wddiff,
                  true_tdump=true_tdump, plume_lons=plume_l, plume_lats=plume_t)

        _plot_gaussian(ax, pp_list, CFG, data_csf, sampled,
                       data_true_sampled, style)

        _plot_flux_vs_time(ax_flux1, CFG, data_csf, nox_result, cems_nox,
                           flux_unit, style,
                           var_to_plot="nox", nox_suffix=suffix_ts)
        _plot_flux_vs_time(ax_flux2, CFG, data_csf, nox_result, cems_nox,
                           flux_unit, style,
                           var_to_plot="etc", nox_suffix=suffix_ts)

        fig.savefig(dfout_path + ".png", dpi=200)
        plt.close("all")

    # --- Primary figure: optimised trajectory (_O) ---
    _build(dfout,
           suffix_map="_H", suffix_ts=CFG["NOX_SUFFIX"],
           traj=tdump, sampled=data_sampled,
           plume_l=plume_lons, plume_t=plume_lats)

    # --- Optional comparison figure: HYSPLIT only (_H) ---
    if CFG.get("o_plot_hysplit", False):
        _build(dfout + "_hysplit",
               suffix_map="_H", suffix_ts="_H",
               traj=tdump, sampled=data_sampled,
               plume_l=None, plume_t=None)

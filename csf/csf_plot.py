import os, sys, glob, IPython, xarray as xr, numpy as np, pandas as pd, geopandas as gpd, pyproj, math, shutil
import matplotlib as mpl, matplotlib.pyplot as plt, contextily as ctx, matplotlib.cm as cmx
from mpl_toolkits.axes_grid1 import make_axes_locatable
sys.path.insert(0, "..")
import CT, FN
# =============================================================================
# PLOT HELPERS
# =============================================================================

def _get_satellite_style(satellite):
	"""Return satellite-specific plot config."""
	if satellite == "trop":
		return dict(voi="no2", flux_var="flux_no2", clr_ortho="white", clr_text="white",
					label=r"TROPOMI NO$\mathregular{_{2}}$ [$\mu$mol m$\mathregular{^{-2}}$]",
					cmap="viridis", pct=(0.00, 0.99), ylim_pad=lambda lo, hi: (0., hi * 1.10))
	if satellite == "oco3":
		return dict(voi="xco2", flux_var="flux_co2", clr_ortho="black", clr_text="black",
					label=r"OCO-3 SAM XCO$\mathregular{_{2}}$ [ppm]",
					cmap="viridis", pct=(0.01, 0.95), ylim_pad=lambda lo, hi: (lo - 0.05, hi + 0.05))


def _fmt_sigpct(val):
	"""Format sigma-percent; cap display at '>100'."""
	return "> 100" if val > 100. else str(val)


def _plot_map(axbig, CFG, target, data, data_sampled, data_true_sampled, data_csf, tdump, sate, style, o_plot_wddiff, true_tdump=None):
	"""Panel 1: basemap + satellite pixels + trajectory + transects."""
	import contextily as ctx
	from mpl_toolkits.axes_grid1 import make_axes_locatable

	#dx, dy = 1.20, 1.20
	dx, dy = 0.50, 0.50
	axbig.set_xlim([target["lon"] - dx, target["lon"] + dx])
	axbig.set_ylim([target["lat"] - dy, target["lat"] + dy])
	for axis in ["x", "y"]:
		axbig.tick_params(axis=axis, which="both", left=False, right=False, bottom=False, top=False, labelleft=False, labelbottom=False)
	ctx.add_basemap(axbig, alpha=1.0, crs="EPSG:4326", attribution=False, source=ctx.providers.OpenStreetMap.Mapnik)
	axbig.scatter(target["lon"], target["lat"], marker="o", color="black")

	divider = make_axes_locatable(axbig)
	cax		= divider.append_axes("bottom", size="2%", pad=0.03)
	voi		= style["voi"]

	# Filter & colormap limits
	if CFG["SATE_INFO"][0] == "trop":
		data = data.loc[data["qa_value"] > 0.5]
	lo = data[voi].sort_values().iloc[int(len(data) * style["pct"][0])]
	hi = data[voi].sort_values().iloc[int(len(data) * style["pct"][1])]
	data.plot(ax=axbig, cax=cax, column=voi, cmap=style["cmap"],
			  alpha=1.0, vmin=lo, vmax=hi, legend=True,
			  legend_kwds={"label": style["label"], "orientation": "horizontal"})

	# Wind direction comparison overlay
	if o_plot_wddiff:
		axbig.plot(data_csf.iloc[:-1]["lon_a3"], data_csf.iloc[:-1]["lat_a3"],
				   color="red", lw=0.7)
		axbig.scatter(data_csf.iloc[:-1]["lon_a3"], data_csf.iloc[:-1]["lat_a3"],
					  edgecolor="red", marker="o", s=3.0, lw=0.7, facecolor="none")
		for i in range(len(data_csf)):
			axbig.plot(data_csf.loc[i, ["lon_a3_i", "lon_a3_f"]].values,
					   data_csf.loc[i, ["lat_a3_i", "lat_a3_f"]].values,
					   color="magenta", lw=0.5)
			if i % 2 == 0:
				axbig.text(data_csf.loc[i, "lon_a3"], data_csf.loc[i, "lat_a3"],
						   str(data_csf.loc[i, "tdump_id"]), fontsize=4)
				axbig.text(data_csf.loc[i, "lon_a3"], data_csf.loc[i, "lat_a3"],
						   str(int(data_csf.loc[i, "wd_diff_avg"])), fontsize=4)

	# Trajectory + transects
	axbig.plot(tdump["longitude"].values, tdump["latitude"].values, color="white", lw=0.8, linestyle="solid", marker="o", markersize=0.1)

	for tid in data_sampled["tdump_id"].unique():
		seg		  = data_sampled.loc[data_sampled["tdump_id"] == tid]
		seg_true  = data_true_sampled.loc[data_true_sampled["tdump_id"] == tid]
		csf_row   = data_csf.loc[data_csf["tdump_id"] == tid].iloc[0]
		a3, a4	  = csf_row["a3"], csf_row["a4"]
		a3_O, a4_O		= csf_row["a3_O"], csf_row["a4_O"]
		fwhm_seg  		= seg.loc[(seg["dist_ortho"] >= a3 - a4/2) & (seg["dist_ortho"] <= a3 + a4/2)]
		fwhm_seg_true  	= seg_true.loc[(seg_true["dist_ortho"] >= a3_O - a4_O/2) & (seg_true["dist_ortho"] <= a3_O + a4_O/2)]

		axbig.plot(seg["lon_ortho"],	  seg["lat_ortho"],		 linestyle="dashed", color=style["clr_ortho"], lw=0.40)
		axbig.plot(fwhm_seg["lon_ortho"], fwhm_seg["lat_ortho"], linestyle="solid", color="red", lw=0.40)
		axbig.plot(fwhm_seg_true["lon_ortho"], fwhm_seg_true["lat_ortho"], linestyle="solid", color="dodgerblue", lw=0.40, zorder=10)
		axbig.scatter(csf_row["lon_a3"], csf_row["lat_a3"], edgecolor="red", marker="o", s=3.0, lw=0.7, facecolor="none")

	sat_label = "TROPOMI" if CFG["SATE_INFO"][0] == "trop" else "OCO-3"
	axbig.text(0.04, 0.96,
			   f"{sat_label}: {pd.to_datetime(sate['time_utc']).strftime('%Y-%m-%d %H:%M')} (UTC)",
			   va="center", transform=axbig.transAxes,
			   color=style["clr_text"], fontsize=7)


def _plot_gaussian_panels(ax, pp_list, CFG, data_csf, data_sampled, style):
	"""Panel 2: per-transect distance vs. NO2 subplots with Gaussian fits."""
	voi		 = style["voi"]
	flux_var = style["flux_var"]
	xlim	 = [-110., 110.]

	# Dynamic y-limits across all transects
	all_vals = pd.concat([data_sampled.loc[data_sampled["tdump_id"] == tid, voi]
						  for tid in data_csf["tdump_id"]])
	lo, hi	 = all_vals.min(), all_vals.max()
	ylim	 = style["ylim_pad"](lo, hi)

	for i, pp in enumerate(pp_list[:len(data_csf)]):
		row = data_csf.loc[i]
		ax[pp].axis("on")

		is_last = (pp == pp_list[min(len(data_csf) - 1, len(pp_list) - 1)])
		ax[pp].tick_params(axis="x", which="both", bottom=True, top=False,
						   labelbottom=is_last, direction="in",
						   left=False, right=False, labelleft=False, labelright=False)
		ax[pp].tick_params(axis="y", which="both", right=True, left=False,
						   labelright=is_last, labelleft=False, direction="in",
						   bottom=False, top=False, labelbottom=False)
		if is_last:
			ax[pp].set_xlabel("Distance [km]")
		ax[pp].set_xlim(xlim)
		ax[pp].set_ylim(ylim)

		# Scatter + Gaussian fit
		tmp = data_sampled.loc[data_sampled["tdump_id"] == row["tdump_id"]].reset_index(drop=True)
		ax[pp].scatter(tmp["dist_ortho"], tmp[voi],
					   marker="o", edgecolor="grey", facecolor="none", s=2., lw=0.4)

		clr    = ("limegreen" if ("QF_gauss_abs" in data_csf.columns and row["QF_gauss_abs"] == 0)
				  else "grey")
		params = row[["a0", "a1", "a2", "a3", "a4"]].values
		x_fit  = np.arange(xlim[0], xlim[1])
		ax[pp].plot(x_fit, FN._gaussian(x_fit, *params), color=clr, lw=1.0)

		x1, x2 = row["a3"] - row["a4"]/2, row["a3"] + row["a4"]/2
		y1, y2	= FN._gaussian(x1, *params), FN._gaussian(x2, *params)
		ax[pp].scatter(row["a3"], (y1 + y2) / 2, marker="o", color="red", s=5.)
		ax[pp].plot([x1, x2], [y1, y2], color="red", lw=0.70)

		# Annotation text
		if not np.isnan(row[flux_var]):
			fs = 4.0
			metrics = [
				(0.85, r"{}".format(int(row["tdump_id"]) + 1)),
				(0.03, f"Flux= {np.round(row[flux_var],2)}"),
				(0.03, fr"$\sigma_{{center}}$= {int(np.round(row['d_center']))}"),
				(0.03, fr"$\sigma_{{left}}$= {int(np.round(row['d_left']))}"),
				(0.03, fr"$\sigma_{{right}}$= {int(np.round(row['d_right']))}"),
				(0.03, fr"$\sigma_{{gap}}$= {int(np.round(row['d_gap']))}"),
				(0.03, fr"r$^2$= {np.round(row['rsq_detrend'], 2)}"),
				(0.03, fr"$\sigma_{{a2}}$= {_fmt_sigpct(np.round(row['a2sig'] / (row['a2'] + 1e-9) * 100.))}"),
				(0.03, fr"$\sigma_{{a4}}$= {_fmt_sigpct(np.round(row['a4sig'] / (row['a4'] + 1e-9) * 100.))}"),
				(0.03, f"MAPE= {np.round(row['mape'], 4)}"),
			]
			for k, (xpos, txt) in enumerate(metrics):
				ax[pp].text(xpos, 1.0 - 0.105 * (k + 1), txt,
							transform=ax[pp].transAxes, fontsize=fs,
							color="black", ha="left")


def _plot_flux_vs_time(axbig2, CFG, data_csf, cems_nox, flux_unit, style):
	"""Panel 3: cross-sectional flux vs. transport time, with optional CEMS overlay."""

	flux_var   = style["flux_var"]
	flux_var_O = flux_var + "_O"
	has_qf	   = "QF_gauss_abs_H" in data_csf.columns  # suffixed now
	qf_col	   = "QF_gauss_abs_H" if has_qf else None

	axbig2.set_xlabel("Transport time [hours]")
	axbig2.set_ylabel(f"Cross-sectional flux {flux_unit}")

	def _scatter_qf(age_col, col, color, marker_good="o", marker_bad="x", label=None):
		if qf_col and qf_col in data_csf.columns:
			axbig2.scatter(data_csf.loc[data_csf[qf_col]==0, age_col],
						   data_csf.loc[data_csf[qf_col]==0, col],
						   marker=marker_good, color=color, label=label, zorder=5)
			axbig2.scatter(data_csf.loc[data_csf[qf_col]==1, age_col],
						   data_csf.loc[data_csf[qf_col]==1, col],
						   marker=marker_bad, color=color, zorder=5)
		else:
			axbig2.scatter(data_csf["age_hours_H"], data_csf[col],
						   marker=marker_good, color=color, label=label, zorder=5)

	# HYSPLIT flux
	age_H = "age_hours_H" if "age_hours_H" in data_csf.columns else "age_hours"
	_scatter_qf(age_H, flux_var + "_H", "dodgerblue", label="CSF (HYSPLIT)")

	# Optimized flux
	age_O = "age_hours_O" if "age_hours_O" in data_csf.columns else age_H
	if flux_var_O in data_csf.columns:
		_scatter_qf(age_O, flux_var_O, "limegreen", label="CSF (Optimized)")

	# NOx lifetime fit (unchanged, uses _H)
	if CFG["SATE_INFO"][0] == "trop" and CFG["o_calc_nox"]:
		_scatter_qf(age_H, "flux_nox_H", "orangered", label="NOx flux (HYSPLIT)")
		x = np.arange(0., 6., 0.1)
		axbig2.plot(x, FN._nox_lifetime(x, data_csf["emis_nox_H"].iloc[0],
										 data_csf["t_pss_H"].iloc[0]), color="orangered")
		axbig2.text(0.5, 0.85,
					f"NOx Emission= {np.round(data_csf.loc[0,'emis_nox_H'],1)} {flux_unit}",
					transform=axbig2.transAxes, fontsize=7)
		axbig2.text(0.5, 0.75,
					f"NOx Lifetime= {np.round(data_csf.loc[0,'nox_lifetime_H'],1)} hrs",
					transform=axbig2.transAxes, fontsize=7)
		if data_csf.loc[0, "t_pss_H"] != -9999.:
			axbig2.axvline(x=data_csf.loc[0, "t_pss_H"], color="grey", linestyle="dotted")

	# CEMS overlay (unchanged)
	if cems_nox is not None and not cems_nox.empty:
		cems_nox   = cems_nox.copy()
		t_ref	   = cems_nox["tstmp_utc"].max()
		cems_nox["transport_hrs"] = (t_ref - cems_nox["tstmp_utc"]).dt.total_seconds() / 3600.
		axbig2.step(cems_nox["transport_hrs"], cems_nox["noxMass_tph_metric"],
					where="mid", color="darkorange", lw=1.2, label="CEMS NOx")
		axbig2.scatter(cems_nox["transport_hrs"], cems_nox["noxMass_tph_metric"],
					   marker="s", color="darkorange", s=18, zorder=5)
		axbig2.text(0.02, 0.92,
					f"CEMS mean= {cems_nox['noxMass_tph_metric'].mean():.2f} {flux_unit}",
					transform=axbig2.transAxes, fontsize=7, color="darkorange")

	axbig2.legend(fontsize=6, loc="upper right")

# In csf_plot.py — update _plot_csf to pass true_tdump down and use suffixed data_csf directly:

def _plot_csf(CFG, target, data, data_sampled, data_true_sampled, data_csf, tdump,
			  sate, flux_unit, o_plot_wddiff, dfout,
			  cems_nox=None, true_tdump=None):
	# --- Figure & axes setup ---
	mpl.rcParams.update(					mpl.rcParamsDefault)
	mpl.rcParams['font.size']=				8
	figsize=								(7.24*1.0, 7.24*0.65)
	ncols_nrows=							(11,7)
	gs_kw=									dict(width_ratios=[1]*ncols_nrows[0], height_ratios=[1]*ncols_nrows[1])			# [1, 1] for equal ratio for two panel
	ss=										[0.01, 0.090, 0.963, 0.99, 0.00, 0.00]
	fig, ax=								plt.subplots(ncols=ncols_nrows[0], nrows=ncols_nrows[1], figsize=figsize, gridspec_kw=gs_kw)
	plt.subplots_adjust(					left=ss[0], bottom=ss[1], right=ss[2], top=ss[3], wspace=ss[4], hspace=ss[5])
	for i in range(0, ncols_nrows[0]):
		for j in range(0, ncols_nrows[1]):
			ax[j,i].axis('off')
	gs=										ax[0, 0].get_gridspec()
	axbig=									fig.add_subplot(gs[0:4, 0:3])
	axbig2=									fig.add_subplot(gs[5:8, 1:11])
	style=									_get_satellite_style(CFG["SATE_INFO"][0])
	pp_list=								[(r, c) for r in range(4) for c in range(3, 11)]	# 4×8 = 32 slots

	# Strip _H suffix for map + gaussian panels (they expect plain column names)
	data_csf_H = data_csf.rename(columns=lambda c: c[:-2] if c.endswith("_H") else c)

	_plot_map(axbig, CFG, target, data, data_sampled, data_true_sampled, data_csf_H, tdump, sate, style, o_plot_wddiff, true_tdump=true_tdump)
	_plot_gaussian_panels(ax, pp_list, CFG, data_csf_H, data_sampled, style)
	_plot_flux_vs_time(axbig2, CFG, data_csf, cems_nox, flux_unit, style)  # full suffixed df

	fig.savefig(dfout + ".png", dpi=200)
	plt.close("all")
	plt.cla()


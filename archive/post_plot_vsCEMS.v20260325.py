import os, sys, glob, IPython, xarray as xr, numpy as np, pandas as pd, math, pyarrow, fnmatch
import matplotlib as mpl, matplotlib.pyplot as plt, contextily as ctx, matplotlib.cm as cmx
from mpl_toolkits.axes_grid1 import make_axes_locatable
from timezonefinder import TimezoneFinder
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, "..")
import CT, FN

# =============================================================================
# PLOT HELPERS
# =============================================================================

def _scatter_panel(ax, df_bin, flux_col, t0, t1):
	"""One transport-time bin panel: TROPOMI flux vs CEMS NOx."""
	x = pd.to_numeric(df_bin["cems_nox_mean"], errors="coerce").values
	y = pd.to_numeric(df_bin[flux_col],		   errors="coerce").values
	mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
	x, y = x[mask], y[mask]
	n = len(x)
	IPython.embed()
	ax.scatter(x, y, s=12, alpha=0.6, edgecolors="steelblue", facecolors="none", lw=0.6)

	if n > 0:
		lim = [0, max(x.max(), y.max()) * 1.15]
		ax.plot(lim, lim, color="grey", lw=0.8, ls="dashed", zorder=0)
		ax.set_xlim(lim); ax.set_ylim(lim)

		if n >= 3:
			m, b  = np.polyfit(x, y, 1)
			r	  = np.corrcoef(x, y)[0, 1]
			x_fit = np.linspace(lim[0], lim[1], 100)
			ax.plot(x_fit, m * x_fit + b, color="tomato", lw=1.0)
			ax.text(0.05, 0.92, f"r={r:.2f}  n={n}",	transform=ax.transAxes, fontsize=5.5)
			ax.text(0.05, 0.82, f"slope={m:.2f}",		 transform=ax.transAxes, fontsize=5.5, color="tomato")

	ax.set_title(f"{t0}–{t1} min  (n={n})", fontsize=6, pad=2)
	ax.tick_params(labelsize=5)
	ax.set_xlabel("CEMS NOx [tNOx/hr]",			 fontsize=5)
	ax.set_ylabel("TROPOMI NO₂ flux [tNO₂/hr]",  fontsize=5)


def _plot_csf_scatter(trop_l2, cems_mean, target, CFG):
	"""One figure (8 panels) per facility: TROPOMI CSF vs CEMS NOx by transport-time bin."""
	flux_col  = CFG["flux_col"]
	bins	  = CFG["transport_bins"]
	base_dout = CT.d_noxno2 + "d_fig/post/" + CFG["CSF_PRCS_VER"] + "/"
	os.makedirs(base_dout, exist_ok=True)

	# Merge CEMS mean onto CSF rows
	merged = trop_l2.merge(cems_mean, on=["facilityId", "source_file"], how="left")
	age_col = 'age_hours'
	merged["transport_min"] = pd.to_numeric(merged[age_col], errors="coerce") * 60.

	for _, trow in target.iterrows():
		tid, tname = trow["facilityId"], trow["facilityName"]
		tname_long = f"{tid}_{tname}".replace(" ", "_")
		df = merged.loc[merged["facilityId"] == tid].reset_index(drop=True)

		if df.empty or flux_col not in df.columns:
			print(f"  [SKIP] {tname_long}: no data or missing '{flux_col}'")
			continue

		fig, axes = plt.subplots(2, 4, figsize=(14, 7))
		axes = axes.flatten()
		fig.subplots_adjust(hspace=0.45, wspace=0.38)

		for ax, (t0, t1) in zip(axes, bins):
			bin_df = df.loc[(df["transport_min"] >= t0) & (df["transport_min"] < t1)]
			_scatter_panel(ax, bin_df, flux_col, t0, t1)

		fig.suptitle(f"{tname} ({tid})	—  TROPOMI CSF vs. CEMS NOx",
					 fontsize=9, y=1.01)

		os.makedirs(base_dout, exist_ok=True)
		fout = os.path.join(base_dout, f"scatter_{tname_long}_{CFG['CSF_PRCS_VER']}.png")
		fig.savefig(fout, dpi=200, bbox_inches="tight")
		plt.close("all")
		print(f"  [SAVED] {fout}")

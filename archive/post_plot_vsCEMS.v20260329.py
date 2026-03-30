# =============================================================================
# post_plot_vsCEMS.py
# =============================================================================
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, "..")
import CT

def _plot_csf_scatter(trop_l2, cems_mean, target, CFG):

	# --- Config ---
	no2_col   = CFG["flux_col"]
	bins	  = CFG["transport_bins"]
	dout	  = CT.d_noxno2 + "d_fig/post/" + CFG["CSF_PRCS_VER"] + "/"
	os.makedirs(dout, exist_ok=True)

	# --- Age column (handles _H / _O / bare suffix) ---
	age_col = next(c for c in ("age_hours_H", "age_hours_O", "age_hours") if c in trop_l2.columns)

	# --- One row per overpass: NOx source (flux_nox_fit at t→0) + NO2 flux ---
	records = []
	for (fid, sf), g in trop_l2.groupby(["facilityId", "source_file"]):
		nox = np.nan
		if "flux_nox_fit" in g.columns:
			v = g.dropna(subset=["flux_nox_fit"])
			if not v.empty:
				nox = float(v.loc[v[age_col].abs().idxmin(), "flux_nox_fit"])
		no2 = np.nan
		if no2_col in g.columns:
			v = g.dropna(subset=[no2_col])
			if not v.empty:
				no2 = float(v.loc[v[age_col].abs().idxmin(), no2_col])
		records.append({"facilityId": fid, "source_file": sf,
						"flux_nox_source": nox, "no2_flux": no2,
						"transport_min": g[age_col].abs().min() * 60.})
	op = pd.DataFrame(records).merge(cems_mean, on=["facilityId","source_file"], how="left")

	# --- Scatter helper ---
	def _panel(ax, x, y, t0, t1, clr, lbl):
		mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
		x, y = x[mask], y[mask]
		n	 = len(x)
		ax.scatter(x, y, s=14, alpha=0.65, edgecolors=clr, facecolors="none", lw=0.7)
		if n > 0:
			hi = max(x.max(), y.max()) * 1.18
			ax.plot([0, hi], [0, hi], color="grey", lw=0.8, ls="dashed")
			#ax.set_xlim([0, hi]); ax.set_ylim([0, hi])
			ax.set_xlim([0, 2.]); ax.set_ylim([0, 2.])
			
			if n >= 3:
				m, b = np.polyfit(x, y, 1)
				r	 = np.corrcoef(x, y)[0, 1]
				ax.plot([0, hi], [m*0+b, m*hi+b], color=clr, lw=1.0, alpha=0.7)
				ax.text(0.05, 0.92, f"r={r:.2f}  n={n}", transform=ax.transAxes, fontsize=5)
				ax.text(0.05, 0.81, f"slope={m:.2f}",	 transform=ax.transAxes, fontsize=5, color=clr)
		ax.set_title(f"{t0}–{t1} min", fontsize=5.5, pad=2)
		ax.tick_params(labelsize=4.5)
		ax.set_xlabel("CEMS NOx [tNOx/hr]", fontsize=4.5)
		ax.set_ylabel(lbl,					fontsize=4.5)

	# --- One figure per facility ---
	for _, trow in target.iterrows():
		tid, tname = trow["facilityId"], trow["facilityName"]
		df = op.loc[op["facilityId"] == tid]
		if df.empty:
			continue

		fig, axes = plt.subplots(2, len(bins), figsize=(2.1*len(bins), 5.2), gridspec_kw={"hspace": 0.55, "wspace": 0.42})
		fig.text(0.005, 0.73, "NOx source\n(PSS, t→0)",    va="center", rotation=90, fontsize=6, color="darkorange", fontweight="bold")
		fig.text(0.005, 0.27, "NO₂ CSF\n(instantaneous)", va="center", rotation=90, fontsize=6, color="steelblue",	fontweight="bold")

		cems_x = pd.to_numeric(df["cems_nox_mean"], errors="coerce").values
		for i, (t0, t1) in enumerate(bins):
			b = df.loc[(df["transport_min"] >= t0) & (df["transport_min"] < t1)]
			cx = pd.to_numeric(b["cems_nox_mean"],	 errors="coerce").values
			_panel(axes[0,i], cx, pd.to_numeric(b["flux_nox_source"], errors="coerce").values, t0, t1, "darkorange", "NOx source [tNOx/hr]")
			_panel(axes[1,i], cx, pd.to_numeric(b["no2_flux"],		  errors="coerce").values, t0, t1, "steelblue",  "NO₂ flux [tNO₂/hr]")

		fig.suptitle(f"{tname} ({tid})	—  TROPOMI vs. CEMS  |	top: NOx source   bottom: NO₂ CSF",
					 fontsize=7, y=1.02)
		fout = os.path.join(dout, f"scatter_{tid}_{tname}_{CFG['CSF_PRCS_VER']}.png".replace(" ","_"))
		fig.savefig(fout, dpi=400, bbox_inches="tight"); plt.close(fig)
		print(f"  [SAVED] {fout}")

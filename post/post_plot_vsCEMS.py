# =============================================================================
# post_plot_vsCEMS.py
#
# One figure per facility:
#	Scatter of NO2 CSF flux (y) vs. CEMS NOx emission rate (x),
#	with each point coloured by transport time (age_hours).
#
# Every point represents one Gaussian fit at one transport step,
# matched to the CEMS record contemporaneous with its emission time.
# =============================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable


def plot_no2_vs_cems(csf, target, CFG, dout):
	"""
	Parameters
	----------
	csf    : DataFrame	 output of post_main._match_cems  (row = one CSF fit)
	target : DataFrame	 facility metadata (facilityId, facilityName, ...)
	CFG    : dict		 post_main CFG
	dout   : str		 output directory for PNGs
	"""
	os.makedirs(dout, exist_ok=True)

	flux_col = CFG['FLUX_COL']		  # 'flux_no2_H'
	cems_col = 'cems_nox_tph'
	age_col  = 'age_hours'

	# Colour map: transport time in hours
	cmap = plt.get_cmap('plasma_r')

	for _, trow in target.iterrows():
		tid   = trow['facilityId']
		tname = trow['facilityName']

		df = csf.loc[
			(csf['facilityId'] == tid)
			& csf[flux_col].notna()
			& csf[cems_col].notna()
		].copy()

		if df.empty:
			print(f'  [PLOT] {tid} {tname}: no matched rows, skipping')
			continue

		x = df[cems_col].values			 # CEMS NOx [metric t/hr]
		y = df[flux_col].values			  # NO2 CSF flux [t NO2/hr]
		t = df[age_col].values			  # transport time [hrs]

		# Finite & positive on both axes
		mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0) & np.isfinite(t)
		x, y, t = x[mask], y[mask], t[mask]
		n = len(x)

		if n == 0:
			print(f'  [PLOT] {tid} {tname}: no finite pairs, skipping')
			continue

		# --- Colour normalisation: 0 to 95th percentile of age ----
		t_max_clr = np.nanpercentile(t, 95)
		norm = mcolors.Normalize(vmin=0, vmax=max(t_max_clr, 0.1))

		# --- Figure ---------------------------------------------------
		fig, ax = plt.subplots(figsize=(5.5, 4.8))
		sc = ax.scatter(x, y, c=t, cmap=cmap, norm=norm, s=18, alpha=0.75, linewidths=0.3, edgecolors='k')
		for i in range(0, len(df)):
			if df.loc[i,'age_hours_H'] <= 0.2:
				text=	ax.text(x[i], y[i], df.loc[i,'time_tag_H'].strftime('%y%m%d'), fontsize=5)

		# 1:1 line and axis limits
		hi = max(np.nanmax(x), np.nanmax(y)) * 1.15
		ax.plot([0, hi], [0, hi], color='grey', lw=0.9, ls='--', label='1:1')
		#ax.set_xlim(0, hi)
		#ax.set_ylim(0, hi)
		#ax.set_xlim(0, 1.6)
		#ax.set_ylim(0, 1.6)

		# Linear fit (log-space is more appropriate here but keep it simple)
		if n >= 3:
			m, b = np.polyfit(x, y, 1)
			r	 = np.corrcoef(x, y)[0, 1]
			xfit = np.array([0, hi])
			ax.plot(xfit, m * xfit + b, color='steelblue', lw=1.2,
					label=f'fit: slope={m:.2f}, r={r:.2f}')
			ax.text(0.97, 0.07,
					f'n = {n}\nslope = {m:.2f}\nr = {r:.2f}',
					transform=ax.transAxes, fontsize=8,
					ha='right', va='bottom',
					bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='grey', alpha=0.8))

		# Colour bar
		cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02)
		cb.set_label('Transport time [hrs]', fontsize=9)

		ax.set_xlabel('CEMS NOx emission rate  [metric t NOx / hr]', fontsize=10)
		ax.set_ylabel('TROPOMI NO₂ CSF flux  [t NO₂ / hr]', fontsize=10)
		ax.set_title(f'{tname}	({tid})\n'
					 f'NO₂ flux (HYSPLIT) vs. contemporaneous CEMS NOx',
					 fontsize=10)
		ax.legend(fontsize=8, loc='upper left')
		ax.tick_params(labelsize=8)

		fig.tight_layout()
		fout = os.path.join(
			dout,
			f'no2_vs_cems_{tid}_{tname}_{CFG["CSF_PRCS_VER"]}.png'.replace(' ', '_')
		)
		fig.savefig(fout, dpi=300, bbox_inches='tight')
		plt.close(fig)
		print(f'  [PLOT] saved → {fout}')

# =============================================================================
# post_main.py	 Post-processing of CSF output	 (compatible with csf_prcs v20260324)
#
# Goal:
#	1. Read all per-facility csf_*.csv files (_read_csf)
#	2. For every individual CSF row (one Gaussian fit at one transport step),
#	   find the CEMS NOx record that corresponds to the time that air parcel
#	   left the stack:
#		   t_emission = t_overpass - age_hours_H
#	   and attach the nearest hourly CEMS value to that row.
#	3. Plot NO2 CSF flux vs. CEMS NOx emission rate, coloured by transport time.
# =============================================================================
import os, sys, fnmatch
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pa_csv
from tqdm import tqdm
from timezonefinder import TimezoneFinder
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "..")
import CT
from post_plot_vsCEMS import plot_no2_vs_cems

# =============================================================================
# CONFIG
# =============================================================================
CFG = {
	'POST_PRCS_VER': CT.PRCS_VER['post'],
	'CSF_PRCS_VER':  CT.PRCS_VER['csf'],
	'TARGET_INFO':	 CT.df_target,

	# Facility IDs to process
	'TARGET_IDS': ['2167'], #'2103', '6076'],

	# CSF column names (suffix _H = HYSPLIT trajectory)
	#'FLUX_COL': 'flux_no2_H',
	'FLUX_COL': 'flux_nox_fit_O',
	'AGE_COL':	'age_hours_O',
	'QF_COL':	'QF_gauss_abs_O',

	# Additional quality filters applied after the primary QF gate.
	# Each tuple: (column, vmin, vmax)
	'QF_EXTRA': [
		('nobs_O',		   50.,   9999.),
		#('u_std_H',		 -9999.,	1.0),
		#('v_std_H',		 -9999.,	1.0),
		('rsq_detrend_O',  0.6,   9999.),
	],

	# CEMS matching: max allowed |t_emission - cems_hour| in hours.
	# CEMS data is hourly so 0.5 h always finds the nearest record exactly;
	# raise to 1.0 h if you want to keep rows where CEMS data has gaps.
	'CEMS_MATCH_TOL_HRS': 0.5,
}

# =============================================================================
# TARGET TABLE
# =============================================================================
target = pd.read_csv(CFG['TARGET_INFO'])
target['facilityId'] = target['facilityId'].astype(str)
target = target.loc[target['facilityId'].isin(CFG['TARGET_IDS'])].reset_index(drop=True)


# =============================================================================
# STEP 1 — READ CSF CSVs
# =============================================================================
def _read_csf(target, CFG):
	"""
	Read all csf_*.csv files for each facility using PyArrow (fast, parallel).
	Returns a long-form DataFrame; one row = one Gaussian fit at one transport step.

	Columns added:
		facilityId, facilityName  — provenance
		source_file				  — CSV filename stem
		t_overpass				  — UTC timestamp of the TROPOMI overpass
		age_hours				  — transport time [hrs] (copy of AGE_COL)
		t_emission				  — t_overpass - age_hours	(when air left stack)
	"""
	base_dir  = CT.d_noxno2 + 'd_dat/csf/' + CFG['CSF_PRCS_VER'] + '/'
	drop_cols = {'lon_a3_i', 'lon_a3_f', 'lat_a3_i', 'lat_a3_f'}

	pa_read  = pa_csv.ReadOptions(use_threads=True)
	pa_parse = pa_csv.ParseOptions(delimiter=',')
	pa_conv  = pa_csv.ConvertOptions(strings_can_be_null=False)

	tasks = []
	for _, row in target.iterrows():
		tid, tname = row['facilityId'], row['facilityName']
		tdir = (base_dir + f'{tid}_{tname}/').replace(' ', '_')
		try:
			tasks += [
				(tid, tname, e.path, e.name)
				for e in os.scandir(tdir)
				if e.is_file()
				and fnmatch.fnmatch(e.name, 'csf_*.csv')
				and '_cems' not in e.name
			]
		except FileNotFoundError:
			print(f'  [WARN] Directory not found: {tdir}')

	print(f'  [READ] {len(tasks)} files across {target["facilityId"].nunique()} facilities')

	def _read_one(tid, tname, path, fname):
		try:
			tbl = pa_csv.read_csv(path, pa_read, pa_parse, pa_conv)
			tbl = tbl.drop([c for c in drop_cols if c in tbl.schema.names])
			for j, f in enumerate(tbl.schema):
				if pa.types.is_timestamp(f.type) and f.type.tz:
					tbl = tbl.set_column(
						j, f.name,
						tbl.column(f.name).cast(pa.timestamp('us', tz='UTC'))
					)
			n = len(tbl)
			for col_name, val in [('source_file', fname),
								   ('facilityId',  tid),
								   ('facilityName', tname)]:
				tbl = tbl.append_column(col_name, pa.array([val] * n))
			return tbl
		except Exception as e:
			print(f'  [WARN] {path}: {e}')
			return None

	with ThreadPoolExecutor(max_workers=16) as ex:
		futures = {ex.submit(_read_one, *t): t for t in tasks}
		tables	= [
			f.result()
			for f in tqdm(as_completed(futures), total=len(futures), desc='Reading CSF')
			if f.result() is not None
		]

	if not tables:
		raise RuntimeError('No CSF files could be read.')

	df = pa.concat_tables(tables, promote_options='default').to_pandas()

	# --- Parse overpass timestamp from filename ----------------------------
	# Filename pattern (csf_prcs v20260324):
	#	csf_S5P_OFFL_L2__NO2____<t_start>_<t_end>_..._<VER>.csv
	# t_start sits at token index 9 after stripping "csf_" prefix and
	# "_<VER>" suffix and splitting on "_".
	ver = CFG['CSF_PRCS_VER']

	def _parse_overpass(fname):
		stem = fname.replace('.csv', '')
		if stem.endswith('_' + ver):
			stem = stem[:-(len(ver) + 1)]
		if stem.startswith('csf_'):
			stem = stem[4:]
		try:
			return pd.to_datetime(stem.split('_')[9], format='%Y%m%dT%H%M%S', errors='coerce')
		except IndexError:
			return pd.NaT

	df['t_overpass'] = df['source_file'].apply(_parse_overpass)

	# --- Emission time per row --------------------------------------------
	# The air parcel sampled by CSF row i left the stack age_hours_H ago.
	age_col = CFG['AGE_COL']
	if age_col in df.columns:
		df['age_hours'] = pd.to_numeric(df[age_col], errors='coerce')
	else:
		print(f'  [WARN] Age column "{age_col}" not found')
		df['age_hours'] = np.nan

	df['t_emission'] = df['t_overpass'] - pd.to_timedelta(df['age_hours'], unit='h')

	print(f'  [READ] {len(df)} total rows loaded')
	return df


# =============================================================================
# STEP 2 — QUALITY FILTER
# =============================================================================
def _apply_qf(df, CFG):
	"""
	Apply the primary Gaussian QF gate (QF_gauss_abs_H == 0) and any
	additional column-level cuts defined in CFG['QF_EXTRA'].
	Also drops rows with no valid flux value or no valid emission time.
	"""
	n0 = len(df)

	# Primary gate
	qf_col = CFG['QF_COL']
	if qf_col in df.columns:
		df = df.loc[df[qf_col] == 0].reset_index(drop=True)
		print(f'  [QF] {qf_col}==0:			{n0} → {len(df)}')
	else:
		print(f'  [QF] "{qf_col}" not found, skipping primary gate')

	# Extra cuts
	for col, vmin, vmax in CFG['QF_EXTRA']:
		if col not in df.columns:
			print(f'  [QF] "{col}" not found, skipping')
			continue
		n_before = len(df)
		vals = pd.to_numeric(df[col], errors='coerce')
		df	 = df.loc[(vals >= vmin) & (vals <= vmax)].reset_index(drop=True)
		print(f'  [QF] {col} [{vmin}, {vmax}]: {n_before} → {len(df)}')

	# Must have a positive flux and a valid emission time
	flux_col = CFG['FLUX_COL']
	n_before = len(df)
	df[flux_col] = pd.to_numeric(df[flux_col], errors='coerce')
	df = df.loc[df[flux_col].notna() & (df[flux_col] > 0)].reset_index(drop=True)
	df = df.loc[df['t_emission'].notna()].reset_index(drop=True)
	print(f'  [QF] valid flux & t_emission: {n_before} → {len(df)}')

	return df


# =============================================================================
# STEP 3 — MATCH CEMS TO EACH CSF ROW
# =============================================================================
def _load_cems(tid, tname, t_min, t_max, target_row, tf):
	"""
	Load CEMS CSV files covering [t_min, t_max], convert local → UTC,
	sum all units per hour, convert lbs/hr → metric t/hr.
	Returns DataFrame with columns [tstmp_utc, cems_nox_tph].
	"""
	tname_long = f'{tid}_{tname}'.replace(' ', '_')
	months = pd.date_range(
		t_min.to_period('M').to_timestamp(),
		t_max.to_period('M').to_timestamp(),
		freq='MS',
	)
	files = [
		os.path.join(CT.d_cems, tname_long,
					 f'{tname_long}_{m.year}_{m.month:02d}.csv')
		for m in months
	]
	files = [f for f in files if os.path.exists(f)]
	if not files:
		print(f'  [CEMS] No files for {tname_long}')
		return pd.DataFrame()

	cems = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

	cems['tstmp_local'] = (pd.to_datetime(cems['date'])
						   + pd.to_timedelta(cems['hour'], unit='h'))
	tz = tf.timezone_at(lat=target_row['lat'], lng=target_row['lon']) or 'UTC'
	cems['tstmp_utc'] = (
		cems['tstmp_local']
		.dt.tz_localize(tz, ambiguous='NaT', nonexistent='NaT')
		.dt.tz_convert('UTC')
		.dt.tz_localize(None)
	)
	cems['noxMass'] = pd.to_numeric(cems['noxMass'], errors='coerce')

	hourly = (cems.groupby('tstmp_utc', as_index=False)['noxMass']
				  .sum()
				  .rename(columns={'noxMass': 'noxMass_tph'}))
	hourly['cems_nox_tph'] = hourly['noxMass_tph'] * 0.000453592   # lbs/hr → metric t/hr

	return hourly[['tstmp_utc', 'cems_nox_tph']].dropna().sort_values('tstmp_utc')


def _match_cems(df, target, CFG):
	"""
	For every CSF row attach the CEMS hourly record whose timestamp is
	closest to t_emission = t_overpass - age_hours, within CEMS_MATCH_TOL_HRS.

	The physical logic:
		The satellite measures a NO2 column at transport age τ hours downstream.
		That air parcel was at the stack at  t_emission = t_overpass - τ.
		The CEMS record for that hour is the relevant emission rate.

	New columns added to df:
		cems_nox_tph   — matched CEMS NOx [metric t NOx / hr]
		cems_t_emit    — CEMS timestamp used for the match
		cems_dt_hrs    — time offset between t_emission and cems_t_emit [hrs]
	"""
	tf	= TimezoneFinder()
	tol = CFG['CEMS_MATCH_TOL_HRS']

	df = df.copy()
	df['cems_nox_tph'] = np.nan
	df['cems_t_emit']  = pd.NaT
	df['cems_dt_hrs']  = np.nan

	for _, trow in target.iterrows():
		tid   = trow['facilityId']
		tname = trow['facilityName']

		fac_mask = df['facilityId'] == tid
		sub = df.loc[fac_mask]
		if sub.empty:
			continue

		# Load CEMS covering the full range of emission times for this facility
		t_min = sub['t_emission'].min() - pd.Timedelta(hours=tol)
		t_max = sub['t_emission'].max() + pd.Timedelta(hours=tol)
		cems = _load_cems(tid, tname, t_min, t_max, trow, tf)
		if cems.empty:
			continue

		cems_t	 = cems['tstmp_utc'].values.astype('datetime64[ns]')
		cems_nox = cems['cems_nox_tph'].values

		for idx in sub.index:
			t_em = df.at[idx, 't_emission']
			if pd.isna(t_em):
				continue
			dt_ns  = np.abs(cems_t - np.datetime64(t_em, 'ns'))
			i_best = int(dt_ns.argmin())
			dt_hrs = float(dt_ns[i_best]) / 3.6e12	 # ns → hours

			if dt_hrs <= tol:
				df.at[idx, 'cems_nox_tph'] = cems_nox[i_best]
				df.at[idx, 'cems_t_emit']  = cems.iloc[i_best]['tstmp_utc']
				df.at[idx, 'cems_dt_hrs']  = dt_hrs

		n_matched = df.loc[fac_mask, 'cems_nox_tph'].notna().sum()
		print(f'  [CEMS] {tid} {tname}: {n_matched} / {fac_mask.sum()} rows matched')

	return df


# =============================================================================
# RUN
# =============================================================================
if __name__ == '__main__':

	# 1. Read all CSF CSV files
	csf = _read_csf(target=target, CFG=CFG)

	# 2. Quality filter
	print (csf)
	csf= 	_apply_qf(csf, CFG)
	csf=	csf.loc[csf[CFG['AGE_COL']]<0.2].reset_index(drop=True)
	csf=	csf.loc[(csf[CFG['FLUX_COL']] >= csf[CFG['FLUX_COL']].quantile(0.00)) & (csf[CFG['FLUX_COL']] <= csf[CFG['FLUX_COL']].quantile(0.98))].reset_index(drop=True)
	# 3. Match each row to its temporally corresponding CEMS record
	csf = _match_cems(csf, target, CFG)

	# 4. Save merged result
	base_dir = CT.d_noxno2 + 'd_dat/csf/' + CFG['CSF_PRCS_VER'] + '/'
	os.makedirs(base_dir, exist_ok=True)
	out_path = base_dir + f'post_csf_cems_{CFG["POST_PRCS_VER"]}.parquet'
	csf.to_parquet(out_path, engine='pyarrow', index=False)
	print(f'  [SAVE] {len(csf)} rows → {out_path}')

	# 5. Plot
	dout = CT.d_noxno2 + 'd_fig/post/' + CFG['CSF_PRCS_VER'] + '/'
	plot_no2_vs_cems(csf=csf, target=target, CFG=CFG, dout=dout)

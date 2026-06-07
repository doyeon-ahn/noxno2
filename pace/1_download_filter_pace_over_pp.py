# =============================================================================
# 1_download_filter_pace_over_pp.py
# Download PACE TRGAS granules for TROPOMI overpasses over New Madrid PP,
# filter to ±3° target box, save clipped .nc, delete raw — file by file.
# =============================================================================
import os, sys, subprocess, time, re, glob
import numpy as np
import pandas as pd
import xarray as xr
sys.path.insert(0, os.pardir)
import CT

## --------------------------------------------------------------------------
## PARAMETERS
## --------------------------------------------------------------------------
CFG = {
	'tropomi_csv':	CT.d_noxno2 + '/d_dat/trop/' + CT.PRCS_VER['trop'] + '/d_csv/' +
					'2167_New_Madrid_Power_Plant_' + CT.PRCS_VER['trop'] + '.csv',
	'd_raw':		CT.d_noxno2 + '/d_dat/pace/' + CT.PRCS_VER['pace'] + '/d_nc_raw/2167_New_Madrid_Power_Plant/',	 # temp; one file at a time
	'd_out':		CT.d_noxno2 + '/d_dat/pace/' + CT.PRCS_VER['pace'] + '/d_nc/2167_New_Madrid_Power_Plant/',
	#'pace_start':	'2024-02-27',
	'pace_start':	'2024-08-29',
	'buffer_deg':	3.0,
	'short_name':	'PACE_OCI_L2_TRGAS',
	'provider':		'OB_CLOUD',
	'cmr_url':		'https://cmr.earthdata.nasa.gov/search/granules.json',
	'KEEP_VARS':    ['total_column_no2', 'total_column_o3', 'quality_no2', 'quality_o3'],
	'complevel':	4,
	'n_retry':		3,
	'f_summary':	'pace_filter_summary_2167.csv',
}
_cems	= pd.read_csv(CT.df_target)							# CT.df_target points to cems_list.csv
_target = _cems.loc[_cems['facilityId'] == 2167].iloc[0]
CFG['target_lat'] = float(_target['lat'])
CFG['target_lon'] = float(_target['lon'])

## --------------------------------------------------------------------------
## HELPERS
## --------------------------------------------------------------------------
import requests

def _cmr_query_date(date_str, cfg):
	"""Return list of granule download URLs from CMR for a given UTC date."""
	params = {
		'short_name':	cfg['short_name'],
		'provider':		cfg['provider'],
		'temporal':		f"{date_str}T00:00:00Z,{date_str}T23:59:59Z",
		'page_size':	500,
		'page_num':		1,
	}
	urls = []
	while True:
		r = requests.get(cfg['cmr_url'], params=params, timeout=30)
		r.raise_for_status()
		hits = r.json()['feed']['entry']
		if not hits:
			break
		for entry in hits:
			for link in entry.get('links', []):
				href = link.get('href', '')
				if href.endswith('.nc') and 'cumulus-prod-public' in href:
					urls.append(href)
					break
		if len(hits) < params['page_size']:
			break
		params['page_num'] += 1
	return urls


def _download_one(url, d_raw, n_retry=3):
	"""Download a single URL via curl ~/.netrc. Returns local path or None on failure."""
	fname = url.split('/')[-1]
	fpath = os.path.join(d_raw, fname)
	for attempt in range(1, n_retry + 1):
		cmd = ['curl', '-f', '-L', '-n',
			   '-c', '/tmp/pace_cookies.txt',
			   '-b', '/tmp/pace_cookies.txt',
			   '-o', fpath, '--', url]
		ret = subprocess.run(cmd, capture_output=True)
		if ret.returncode == 0:
			return fpath
		print(f"	  attempt {attempt}/{n_retry} failed (rc={ret.returncode})")
		if attempt < n_retry:
			time.sleep(5)
	return None


def _pace_timestamp(fname):
	"""Parse PACE_OCI.YYYYMMDDTHHMMSS.*.nc -> pd.Timestamp (tz-naive UTC)."""
	m = re.search(r'PACE_OCI\.(\d{8}T\d{6})\.', fname)
	return pd.Timestamp(m.group(1)) if m else pd.NaT


def _clip_and_save(fpath_raw, t_pace, df_trop, cfg):

	"""
	Clip raw PACE file to target box, match nearest TROPOMI row,
	save filtered .nc. Returns summary dict.
	"""
	lat_c, lon_c, buf = cfg['target_lat'], cfg['target_lon'], cfg['buffer_deg']
	fname = os.path.basename(fpath_raw)

	with xr.open_dataset(fpath_raw, group='geophysical_data', engine='netcdf4') as bb, \
		 xr.open_dataset(fpath_raw, group='navigation_data',  engine='netcdf4') as cc:

		lat = cc['latitude'].values
		lon = cc['longitude'].values

		mask = ((lat > lat_c - buf) & (lat < lat_c + buf) &
				(lon > lon_c - buf) & (lon < lon_c + buf)).squeeze()

		if not mask.any():
			return {'pace_file': fname, 't_pace': str(t_pace),
					'overlap': False, 'n_pts': 0,
					'latmin': np.nan, 'latmax': np.nan,
					'lonmin': np.nan, 'lonmax': np.nan,
					'trop_file': '', 't_trop_mid': '', 'dt_min': np.nan}

		ii, jj	 = np.where(mask)
		si0, si1 = int(ii.min()), int(ii.max() + 1)
		pj0, pj1 = int(jj.min()), int(jj.max() + 1)

		bb_sub = bb.isel(number_of_lines=slice(si0, si1), pixels_per_line=slice(pj0, pj1))
		cc_sub = cc.isel(number_of_lines=slice(si0, si1), pixels_per_line=slice(pj0, pj1))

		existing = [v for v in cfg['KEEP_VARS'] if v in bb_sub]
		if not existing:
			print(f"	 WARNING: none of KEEP_VARS found in file; keeping all geophysical vars")
			print(f"	 available vars: {list(bb_sub.data_vars)}")
			existing = list(bb_sub.data_vars)
		ds = xr.merge([bb_sub[existing], cc_sub[['latitude', 'longitude']]]).load()

	## Downcast float64 -> float32
	for v in ds.data_vars:
		if ds[v].dtype == np.float64:
			ds[v] = ds[v].astype(np.float32)

	lat_v = ds['latitude'].values.ravel()
	lon_v = ds['longitude'].values.ravel()
	n_pts = int(mask[si0:si1, pj0:pj1].sum())

	## Nearest TROPOMI overpass
	dt_abs		 = (df_trop['t_mid'] - t_pace).abs()
	idx			 = dt_abs.idxmin()
	t_trop_mid	 = df_trop.loc[idx, 't_mid']
	trop_file	 = df_trop.loc[idx, 'file']
	dt_min		 = (t_pace - t_trop_mid).total_seconds() / 60.0

	## Save filtered .nc
	f_out	 = fname.replace('.nc', '_2167_NMadrid_filtered.nc')
	out_path = os.path.join(cfg['d_out'], f_out)
	encoding = {v: {'zlib': True, 'complevel': cfg['complevel'], 'shuffle': True} for v in ds.data_vars}
	ds.to_netcdf(out_path, encoding=encoding)

	return {'pace_file':  f_out,
			't_pace':	  str(t_pace),
			'overlap':	  True,
			'n_pts':	  n_pts,
			'latmin':	  float(np.nanmin(lat_v)), 'latmax': float(np.nanmax(lat_v)),
			'lonmin':	  float(np.nanmin(lon_v)), 'lonmax': float(np.nanmax(lon_v)),
			'trop_file':  trop_file,
			't_trop_mid': str(t_trop_mid),
			'dt_min':	  dt_min}


## --------------------------------------------------------------------------
## MAIN
## --------------------------------------------------------------------------

def main(cfg):
	## Load TROPOMI table; build tz-naive mid-time
	df_trop = pd.read_csv(cfg['tropomi_csv'])
	df_trop['t_mid'] = df_trop.apply(
		lambda r: pd.Timestamp(r['time_utc_i'].replace('Z','')) +
				  (pd.Timestamp(r['time_utc_f'].replace('Z','')) -
				   pd.Timestamp(r['time_utc_i'].replace('Z',''))) / 2,
		axis=1)

	## Filter to PACE era; get unique dates
	df_trop = df_trop.loc[df_trop['t_mid'] >= pd.Timestamp(cfg['pace_start'])].reset_index(drop=True)
	dates	= sorted(df_trop['t_mid'].dt.date.unique())
	print(f"TROPOMI rows in PACE era: {len(df_trop)}  |  unique dates: {len(dates)}")

	os.makedirs(cfg['d_raw'], exist_ok=True)
	os.makedirs(cfg['d_out'], exist_ok=True)

	## Load existing summary to allow resuming
	summary_path = os.path.join(cfg['d_out'], cfg['f_summary'])
	if os.path.isfile(summary_path):
		df_done  = pd.read_csv(summary_path)
		done_raw = set(
			r.replace('_2167_NMadrid_filtered.nc', '.nc')
			for r in df_done['pace_file'].tolist()
		)
		summary  = df_done.to_dict('records')
		print(f"Resuming — {len(done_raw)} granules already processed")
	else:
		done_raw = set()
		summary  = []

	## Main loop: date -> CMR -> download -> clip -> delete raw
	for i_d, d in enumerate(dates):
		date_str = str(d)
		print(f"\n[{i_d+1}/{len(dates)}] {date_str}", end='  ', flush=True)

		try:
			urls = _cmr_query_date(date_str, cfg)
		except Exception as e:
			print(f"CMR error: {e}")
			continue
		print(f"{len(urls)} granules")

		for url in urls:
			fname = url.split('/')[-1]

			## Skip if already processed
			if fname in done_raw:
				print(f"  skip (done): {fname}")
				continue

			print(f"  -> {fname}")

			## Download
			fpath_raw = _download_one(url, cfg['d_raw'], cfg['n_retry'])
			if fpath_raw is None:
				print(f"	 FAILED download, skipping")
				summary.append({'pace_file': fname, 't_pace': '', 'overlap': False,
								'n_pts': 0, 'latmin': np.nan, 'latmax': np.nan,
								'lonmin': np.nan, 'lonmax': np.nan,
								'trop_file': '', 't_trop_mid': '', 'dt_min': np.nan,
								'status': 'download_failed'})
				continue

			## Parse timestamp from filename
			t_pace = _pace_timestamp(fname)

			## Clip, match TROPOMI, save filtered .nc
			try:
				row = _clip_and_save(fpath_raw, t_pace, df_trop, cfg)
				row['status'] = 'ok'
				overlap_str = f"n_pts={row['n_pts']}  dt={row['dt_min']:+.1f} min" if row['overlap'] else "no overlap"
				print(f"	 {overlap_str}")
			except Exception as e:
				print(f"	 clip error: {e}")
				row = {'pace_file': fname, 't_pace': str(t_pace), 'overlap': False,
					   'n_pts': 0, 'latmin': np.nan, 'latmax': np.nan,
					   'lonmin': np.nan, 'lonmax': np.nan,
					   'trop_file': '', 't_trop_mid': '', 'dt_min': np.nan,
					   'status': 'clip_error'}

			## Delete raw file regardless
			os.remove(fpath_raw)

			summary.append(row)
			done_raw.add(fname)

			## Persist summary after every file (safe resume)
			pd.DataFrame(summary).to_csv(summary_path, index=False)

	## Final report
	df_s = pd.DataFrame(summary)
	print(f"\nDone.  Total={len(df_s)}	overlap={df_s['overlap'].sum()}  "
		  f"no_overlap={(~df_s['overlap']).sum()}  "
		  f"errors={( df_s['status'] != 'ok').sum()}")
	print(f"Summary: {summary_path}")
	return df_s


if __name__ == '__main__':
	main(CFG)

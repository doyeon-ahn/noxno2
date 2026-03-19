from pathlib import Path
import numpy as np
import shutil, json
from typing import Any
"""
def _SNAPSHOT(tag: str, scripts: list[str]) -> dict:
def _FILENAME(data, p_data):
"""

def _SNAPSHOT(name: str, tag: str, scripts: list[str], cfg: dict[str, Any], out_dir: str) -> dict:
	"""
	Snapshot code + runtime config into one place.
	- Copies each script to <out_dir>/<scriptstem>__<tag>.py
	- Writes config to <out_dir>/<name>__<tag>.json where name defaults to the first script stem.
	"""
	out_dir= Path(out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	for s in scripts:
		src = Path(s)
		if not src.exists():
			raise FileNotFoundError(src)
		shutil.copy2(src, out_dir / f"{src.stem}__{tag}{src.suffix}")
	cfg_path = out_dir / f"{name}__{tag}.json"
	cfg_path.write_text(json.dumps(cfg, indent=2))

def _FILENAME(data, p_data):
	if data == 'oco3':
		return f"oco3_LtCO2_{p_data['YYMMDD']}_{p_data['VERSION']}_*.nc4"	# 'oco3_LtCO2_{YYMMDD}_{version}_*.nc4'
	if data == 'trop':
		return f"S5P_{p_data['processingMode']}_L2__NO2____{p_data['YYYYMMDD']}T*_*_{p_data['orbitID']}_*_*_*.nc_keyvars.nc"
	if data == 'pace':
		return f"PACE_OCI.{p_data['YYMMDD']}T*.L2.TRGAS.{p_data['VERSION']}.nc"
	if data == 'ghssmod':
		return [f"GHS_SMOD_{p_data['epoch']}_GLOBE_{p_data['release']}_54009_1000_{p_data['version']}.tif",\
				f"GHS_SMOD_{p_data['epoch']}_GLOBE_{p_data['release']}_54009_1000_UC_{p_data['version']}.shp"]

def _func_nox_lifetime(t, Q, lifetime):
	return (Q * np.exp(-1.*t/lifetime))

def _gaussian(x, a0, a1, a2, a3, a4):
	return (	a0 + (a1*x) + (a2*np.exp(-4.*np.log(2)*((x-a3)**2) * (a4**-2)))		)							# Reuter et al. (2019)


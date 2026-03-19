import os, sys, glob, IPython, xarray as xr, numpy as np, pandas as pd
sys.path.insert(0, "..")
import CT, FN
"""
Read and process TROPOMI .nc files
"""
## PARAMETERS
CFG=							{	'TROP_PRCS_VER':	CT.PRCS_VER["trop"],
									'TROP_DATA_VER':	CT.DATA_VER["trop"],
									'target_list':		CT.df_target,
									'processingMode':	'*',				# 'OFFL', 'RPRO', '*'
									'YYYYMMDD':			'*',
									'orbitID':			'*',
									'sfilter_buffer':	2.,					# [degrees]; i.e., lon_mid - sfilter_buffer ~ lon_mid + sfilter_buffer
									'KEEP_VARS':		[	"latitude", "longitude", "latitude_bounds", "longitude_bounds", "time_utc",
															"nitrogendioxide_tropospheric_column",
															"nitrogendioxide_tropospheric_column_precision",
															"qa_value",
															"air_mass_factor_total",
															"air_mass_factor_troposphere",
															"averaging_kernel",
															"tm5_constant_a", "tm5_constant_b", "tm5_tropopause_layer_index",
															"solar_zenith_angle",
															"surface_altitude", "surface_pressure",
															"water_slant_column_density",
															"water_liquid_slant_column_density" ],
									'o_save_clipping_csv':	True,
									'o_save_data_nc':		False,
									'snapshot':				True}

target=							pd.read_csv(CFG['target_list'])
target['facilityId']=			target['facilityId'].astype('str')
target=							target.loc[target['facilityId'].isin(['6705', '6076', '8102', '6165', '6481', '6002', '2103', '6146', '2832', '2823', '2167', '2168'])].reset_index(drop=True)

## Functions
def _save_clipping_coord(file, target_row, ds_sub, scanline_i0, scanline_i1, pixel_j0, pixel_j1, d_out, f_out):
	os.makedirs(d_out, exist_ok=True)
	row = {
		"file":		   os.path.basename(file),
		"n_data":	   int(ds_sub["nitrogendioxide_tropospheric_column"].notnull().sum().item()),
		"time_utc_i":  ds_sub["time_utc"].min().item(),
		"time_utc_f":  ds_sub["time_utc"].max().item(),
		"latmin":	   float(ds_sub["latitude"].min().item()),
		"latmax":	   float(ds_sub["latitude"].max().item()),
		"lonmin":	   float(ds_sub["longitude"].min().item()),
		"lonmax":	   float(ds_sub["longitude"].max().item()),
		"scanline_i0": int(scanline_i0),
		"scanline_i1": int(scanline_i1),
		"pixel_j0":    int(pixel_j0),
		"pixel_j1":    int(pixel_j1),
	}
	pd.DataFrame([row]).to_csv(d_out+f_out, mode="a", header=not os.path.exists(d_out+f_out), index=False)

def _trop_prcs(CFG, target, d_in):
	tname=						(target["facilityId"]+'_'+target["facilityName"]).str.replace(' ', '_').values
	latmin=						target["lat"].values - CFG['sfilter_buffer']
	latmax=						target["lat"].values + CFG['sfilter_buffer']
	lonmin=						target["lon"].values - CFG['sfilter_buffer']
	lonmax=						target["lon"].values + CFG['sfilter_buffer']

	## Read .nc files
	files=						sorted(glob.glob(d_in + FN._FILENAME(data='trop', p_data={'processingMode':CFG['processingMode'], 'YYYYMMDD':CFG['YYYYMMDD'], 'orbitID':CFG['orbitID']})))
	for i_file, file in enumerate(files):
		file=	files[i_file]
		print(f"	Processing ... {i_file}/{len(files)}: {file}")

		## Open lazily (fast). Only force-load lat/lon arrays for masking.
		with xr.open_dataset(file) as ds:
			lat=				ds["latitude"].values		
			lon=				ds["longitude"].values	 
			lat_lo, lat_hi =	np.nanmin(lat), np.nanmax(lat)
			lon_lo, lon_hi =	np.nanmin(lon), np.nanmax(lon)
			## Loop each target
			for i in range(len(tname)):
				## Quick global reject: if a target bbox doesn't overlap file extent, skip mask calc (wrap-aware lon overlap)
				if (latmax[i] < lat_lo) or (latmin[i] > lat_hi):
					continue

				## mask: overlap between target box and TROPOMI swath
				lon_mask = ((lon > lonmin[i]) & (lon < lonmax[i])) if (lonmin[i] <= lonmax[i]) else ((lon > lonmin[i]) | (lon < lonmax[i]))	# (aware of dateline-crossing case)
				mask = ((lat > latmin[i]) & (lat < latmax[i]) & lon_mask).squeeze()
				if not mask.any():
					continue

				## Find minimal rectangle in (scanline, ground_pixel) index space
				ii, jj=						np.where(mask)
				scanline_i0, scanline_i1=	int(ii.min()), int(ii.max()+1)
				pixel_j0, pixel_j1=			int(jj.min()), int(jj.max()+1)
				ds_sub=						ds.isel(scanline=slice(scanline_i0, scanline_i1), ground_pixel=slice(pixel_j0, pixel_j1))
				
				## Save clipping cooridates to .csv
				if CFG['o_save_clipping_csv'] == True:
					_save_clipping_coord(	file=file, target_row=target.iloc[i], ds_sub=ds_sub, scanline_i0=scanline_i0, scanline_i1=scanline_i1, pixel_j0=pixel_j0, pixel_j1=pixel_j1,
											d_out=CT.d_trop_target+'/'+CFG['TROP_PRCS_VER']+'/d_csv/', f_out=f"{tname[i]}_{CFG['TROP_PRCS_VER']}.csv")

				## Save clipped data to .nc
				if CFG['o_save_data_nc'] == True:
					## Keep only selected variables (silently ignore missing ones)
					ds_sub=			ds_sub[CFG['KEEP_VARS']]
					## Downcast float64 -> float32 to shrink size (keep ints/bools as-is)
					for v in ds_sub.data_vars:
						if np.issubdtype(ds_sub[v].dtype, np.floating) and ds_sub[v].dtype == np.float64:
							ds_sub[v]= ds_sub[v].astype(np.float32)
					## Compressed NetCDF encoding (zlib) 
					encoding=		{v: {"zlib": True, "complevel": 4, "shuffle": True} for v in ds_sub.data_vars}
					## preserve your existing choice for time_utc fill
					if "time_utc" in ds_sub.variables:
						encoding["time_utc"] = {"_FillValue": None}
					## Save
					d_out=			CT.d_trop_target+'/'+CFG['TROP_PRCS_VER']+'/d_nc/'+tname[i]+'/'
					if os.path.isdir(d_out) == False:
						os.makedirs(d_out)
					f_out=			f"{file.split('/')[-1]}_{tname[i]}_{CFG['TROP_PRCS_VER']}.nc"
					ds_sub.to_netcdf(d_out + f_out, encoding=encoding)

## Run
_=					_trop_prcs(CFG=CFG, target=target, d_in=CT.d_trop)

## History
if CFG['snapshot'] == 'True':
	TAG=			f"{CFG['TROP_PRCS_VER']}"
	_=				FN._SNAPSHOT(name="trop_prcs", tag=TAG, scripts=['trop_prcs.py'], cfg=CFG, out_dir=os.path.join(CT.d_satcsf, "d_history"))


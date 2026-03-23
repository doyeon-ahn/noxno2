import os, sys, glob, IPython, xarray as xr, numpy as np, pandas as pd, geopandas as gpd, pyproj, math, shutil
from netCDF4 import Dataset
from timezonefinder import TimezoneFinder
from datetime import timedelta, date
sys.path.insert(0, "..")
import CT, FN

## Parameters ____
CFG=	{	'HYSTRAJ_RUN_VER':			CT.PRCS_VER["hystraj"],
			'target_list':				CT.df_target,
			'traj_starting_time':		['trop', CT.DATA_VER["trop"], CT.PRCS_VER["trop"], '2019-01-01', '2025-12-31'], 
			'TROP_PRCS_VER':			CT.PRCS_VER['trop'],
			'traj_mode':				'plume',	# 'plume': Simulate urban plume (forward); 'bg': Simulate background area (backward)
			'o_make_emitime':			False,		# default: False
			'nhrs_prior_ctr':			[-2, 6],	# [start traj (or emissions) from nhrs back from the earlier overpass time, finish traj (or emissions) N hour after the later overpass time]
			'alt_esource':				150.,		# Typical large coal or oil-fired power plant stacks generally range from ~100–200+ meters
			'd_met':					CT.d_era5,
			'o_use_SETUP':				True,
			'o_run_hyts_std':			True,
			'o_skip_if_tdump_exist':	False,
			'snapshot':					True	}

target=						pd.read_csv(CFG['target_list'])
target['facilityId']=      	target['facilityId'].astype('str')
target=      				target.loc[target['facilityId'].isin(['6705', '6076', '8102', '6165', '6481', '6002', '2103', '6146', '2832', '2823', '2167', '2168'])].reset_index(drop=True)


def _hysplit_write_setup():
	d_setup= CT.d_hysplit+'working/'
	f_setup= 'SETUP.CFG'
	o= open(d_setup+f_setup, "w")
	o.write(' &SETUP'+'\n')
	#o.write('tratio = 0.75,'+'\n')
	#o.write('delt = 0.0,'+'\n')
	#o.write('mgmin = 10,'+'\n')
	#o.write('khmax = 9999,'+'\n')
	#o.write('kmixd = 0,'+'\n')
	#o.write('kmsl = 0,'+'\n')
	#o.write('kagl = 1,'+'\n')
	#o.write('k10m = 1,'+'\n')
	#o.write('nstr = 0,'+'\n')
	#o.write('mhrs = 9999,'+'\n')
	#o.write('nver = 0,'+'\n')
	o.write('tout = 15,'+'\n')
	#o.write('tm_pres = 1,'+'\n')
	#o.write('tm_tpot = 0,'+'\n')
	#o.write('tm_tamb = 0,'+'\n')
	#o.write('tm_rain = 0,'+'\n')
	#o.write('tm_mixd = 0,'+'\n')
	#o.write('tm_relh = 0,'+'\n')
	#o.write('tm_sphu = 0,'+'\n')
	#o.write('tm_mixr = 0,'+'\n')
	#o.write('tm_dswf = 0,'+'\n')
	#o.write('tm_terr = 0,'+'\n')
	#o.write('tm_uwnd = 0,'+'\n')
	#o.write('tm_vwnd = 0,'+'\n')
	#o.write('dxf = 1.00,'+'\n')
	#o.write('dyf = 1.00,'+'\n')
	#o.write('dzf = 0.01,'+'\n')
	#o.write('messg = 'MESSAGE','+'\n')
	o.write(" VARSIWANT = 'TEMP', 'SIGW', "+'\n')
	o.write('/'+'\n')

def _hystraj_make_control(CFG, target):
	## Loop each target
	for it in range(0, len(target)): 
		print							(f"	Processing ... {it}/{len(target)}: {target.loc[it,'facilityId']}, {target.loc[it,'facilityName']}")
		tname=							(target.loc[it,"facilityId"]+'_'+target.loc[it,"facilityName"]).replace(' ', '_')

		## Read satellite observation time
		if CFG['traj_starting_time'][0] == 'trop':
			d_sate=						CT.d_trop_target+CFG['TROP_PRCS_VER']+'/d_nc/'+tname+'/'
			f_sate=						sorted(glob.glob(d_sate + f"*_{tname}_{CFG['TROP_PRCS_VER']}.nc"))
			time_utc_list= 				[]
			for f in f_sate:
				with Dataset(f, 'r') as ds:
					time_val=			ds.variables['time_utc'][0,0]  # first entry
					time_utc_list.append(time_val)
			sate= 						pd.DataFrame({'time_utc': time_utc_list})
			sate['time_utc']=			pd.to_datetime(sate['time_utc'])#, format='ISO8601')
			sate=						sate[(sate['time_utc'] >= CFG['traj_starting_time'][3]) & (sate['time_utc'] <= CFG['traj_starting_time'][4])].reset_index(drop=True)

		# Define dir names
		d_cdump=						CT.d_noxno2+'d_dat/hysplit/'+CFG['HYSTRAJ_RUN_VER']+'/'+tname+'/cdump/'
		d_tdump=						CT.d_noxno2+'d_dat/hysplit/'+CFG['HYSTRAJ_RUN_VER']+'/'+tname+'/tdump/'
		d_control=						CT.d_noxno2+'d_dat/hysplit/'+CFG['HYSTRAJ_RUN_VER']+'/'+tname+'/control/'
		d_et=							CT.d_noxno2+'d_dat/hysplit/'+CFG['HYSTRAJ_RUN_VER']+'/'+tname+'/EMITIMES/'
		if os.path.isdir(d_cdump) == False:
			os.makedirs(d_cdump)
		if os.path.isdir(d_cdump+'d_cdumpasc/') == False:
			os.mkdir(d_cdump+'d_cdumpasc/')
		if os.path.isdir(d_tdump) == False:
			os.makedirs(d_tdump)
		if os.path.isdir(d_control) == False:
			os.makedirs(d_control)
		if os.path.isdir(d_et) == False:
			os.makedirs(d_et)

		# Write SETUP file
		if CFG['o_use_SETUP'] == True:
			_hysplit_write_setup()

		# Write CONTROL for each overpass 
		for i_sate in range(0, len(sate)):
			# Define filenames
			fname_suffix=				CFG['traj_starting_time'][0]+'_'+target.loc[it,'facilityId']+'_'+CFG['traj_mode']+'_'+sate.loc[i_sate,'time_utc'].strftime('%Y%m%d%H%M')+'_'+CFG['HYSTRAJ_RUN_VER']
			f_cdump=					'cdump_'+fname_suffix
			f_tdump=					'tdump_'+fname_suffix
			f_control=					'CONTROL_'+fname_suffix
			f_et=						'EMITIMES_'+fname_suffix

			# Traj starting location
			lat_esource=					[target.loc[it,'lat']]
			lon_esource=					[target.loc[it,'lon']]
			alt_esource=					[CFG['alt_esource']]

			# Determine emission cycle
			if CFG['traj_mode'] == 'plume':
				time_start=					sate.loc[i_sate,'time_utc'] + pd.Timedelta(CFG['nhrs_prior_ctr'][0], 'hour')		# start emissions from nhrs back from the earlier overpass time
				time_end=					sate.loc[i_sate,'time_utc'] + pd.Timedelta(CFG['nhrs_prior_ctr'][1], 'hour')		# finish emissions N hour after the later overpass time

			time_start_ctr=					str(time_start.year)[2:4]+' '+str(time_start.month).zfill(2)+' '+\
											str(time_start.day).zfill(2)+' '+str(time_start.hour).zfill(2)+' '+str(time_start.minute).zfill(2)				# Emission start time (and HYSPLIT start time)
			nhrs_duration_ctr=				str(math.ceil((time_end - time_start).total_seconds() / 3600.))						# Emission duration

			if CFG['o_make_emitime'] == False:
				ndata_ctr=					len(lon_esource)

			tstmp_tmp=						time_start
			if CFG['o_make_emitime'] == True:
				ndata_ctr=					0
				while tstmp_tmp < time_end:
					ndata_ctr=				ndata_ctr + len(lon_esource)
					tstmp_tmp=				tstmp_tmp + pd.Timedelta(emission_gap_minute, 'minute')

			# Determine metfiles
			time_end_extra1hr=				sate.loc[i_sate,'time_utc'] + pd.Timedelta(CFG['nhrs_prior_ctr'][1]+1, 'hour')		# adding extra 1 hour because ERA5 covers 23:xx
			metdate_start=					date(time_start.year, time_start.month, time_start.day)
			metdate_end=					date(time_end_extra1hr.year, time_end_extra1hr.month, time_end_extra1hr.day)
			if CFG['traj_mode'] == 'plume':
				metdate_list=				pd.date_range(start=metdate_start, end=metdate_end)
			if CFG['traj_mode'] == 'bg':
				metdate_list=				pd.date_range(start=metdate_end, end=metdate_start)
			nmet_ctr=						str(len(metdate_list))
			dmet_ctr=						[CFG['d_met']] * len(metdate_list)
			fmet_ctr=						[]
			for i in range(0, len(metdate_list)):
				fmet_ctr.append('ERA5_'+str(metdate_list[i].year)+str(metdate_list[i].month).zfill(2)+str(metdate_list[i].day).zfill(2)+'_global.ARL')

			# Etc
			if os.path.isfile(d_tdump+f_tdump) == False:
				print						('		Output tdump file ('+d_tdump+f_tdump+') not found. Continue running .. '+CFG['traj_starting_time'][0]+', '+str(i_sate)+'/'+str(len(sate)))
			if os.path.isfile(d_tdump+f_tdump) == True:
				if CFG['o_skip_if_tdump_exist'] == 0:
					print					('		Output tdump file exist ('+d_tdump+f_tdump+'), Overwriting .. '+CFG['traj_starting_time'][0]+', '+str(i_sate)+'/'+str(len(sate)))
				if CFG['o_skip_if_tdump_exist'] == 1:
					print					('		Output tdump file exist ('+d_tdump+f_tdump+'), Stop running .. '+CFG['traj_starting_time'][0]+', '+str(i_sate)+'/'+str(len(sate)))
					sys.exit()

			# Write CONTROL file
			o= open(d_control+f_control, "w")
			o.write(time_start_ctr+'\n')									# 1) starting time, yy mm dd hh mm

			o.write(str(ndata_ctr)+'\n')									# 2) number of starting locatoins; In the CONTROL file you will need to have as many sources lines as you have in the EMITIMES file.
			for j in range(0, len(lon_esource)):
				o.write(str(lat_esource[j])+' '+str(lon_esource[j])+' '+str(alt_esource[j])+'\n')	# 3) starting locations; Lat, Lon, Alt(m agl), Fixed to single value bc this parameter is meaningless when EMITIMES exists
			o.write(nhrs_duration_ctr+'\n')									# 4) total run time ; HYSPLIT RUN TIME IN HRS
			o.write('0\n')													# 5) VERTICAL MOTION METHOD
			o.write('10000.0\n')											# 6) TOP OF MODEL
			o.write(nmet_ctr+'\n')											# 7) # of input data grids
			for j in range(0, len(fmet_ctr)):
				o.write(dmet_ctr[j]+'\n')									# 8) Meteorological data grid # 1 directory
				o.write(fmet_ctr[j]+'\n')									# 9) Meteorological data grid # 1 file name
			o.write(d_tdump+'\n')											# 10) Directory of trajectory output file
			o.write(f_tdump+'\n')											# 11) Name of the trajectory endpoints file
			o.close()

			# Copy CONTROL file to working dir
			shutil.copy(d_control+f_control, CT.d_hysplit+'working/CONTROL')

			# Run hyts_std
			if CFG['o_run_hyts_std'] == True:
				os.chdir(CT.d_hysplit+'working')												# Change directory
				if CFG['o_use_SETUP'] == False and os.path.isfile('./SETUP.CFG'):
					os.remove('SETUP.CFG')														# Comment this line if SETUP.CFG is not needed (SETUP.CFG is manually defined in /working dir)
				os.system(CT.d_hysplit+'exec/hyts_std')											# Run hysplit (make sure SETUP.CFG from particle dispersion run does not exist in working dir!)
				os.chdir(CT.d_code+'hys_traj/')													# Change directory


## Run
_=				_hystraj_make_control(CFG=CFG, target=target)


## History
if CFG['snapshot'] == True:
    TAG=            f"{CFG['HYSTRAJ_RUN_VER']}"
    _=              FN._SNAPSHOT(name="hystraj_run", tag=TAG, scripts=['hystraj_run.py'], cfg=CFG, out_dir=os.path.join(CT.d_noxno2, "d_history"))



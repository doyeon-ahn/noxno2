import glob, os
import pandas as pd


# Parameters
"""
metdate_list <-- List of dates (pandas.core.indexes.datetimes.DatetimeIndex)
			 	 This is defined in hys_traj_make_control.py, but could be manually defined in this program if needed
"""

# Geographical boundary
lon_upperleft=								-180.
lat_upperleft=								-50.
lon_lowerright=								180.
lat_lowerright=								60.

area=										str(round(lat_upperleft,2))+'/'+str(round(lon_upperleft,2))+'/'+str(round(lat_lowerright,2))+'/'+str(round(lon_lowerright,2))

#d_met=										'/Users/dyahn/data/era5/'
d_met=										'/Volumes/ic/era5_global/'
metdate_list=								pd.date_range(start='2025-02-16', end='2025-12-01', freq='1D')

d_era52arl=									'/Users/dahn/hysplit_data2arl/era52arl/'

for ii in range(0, len(metdate_list)):
	# Get ERA5
	year=									str(metdate_list[ii].year)
	month=									str(metdate_list[ii].month)
	day=									str(metdate_list[ii].day)

	# Check whether metfile already exist in local machine
	metfile_tmp=							'ERA5_'+str(metdate_list[ii].year)+str(metdate_list[ii].month).zfill(2)+str(metdate_list[ii].day).zfill(2)+'_global.ARL'
	metfile=								glob.glob(d_met + metfile_tmp)

	if len(metfile) == 1:
		print								('		Met file '+ metfile_tmp + ' found. Skipping download.')

	if len(metfile) == 0:
		print								('		Met file '+ metfile_tmp + ' NOT found. Ddownload ... ')
		# Download era5
		os.system(							'python get_era5_cds.py --3d -y ' +year+' -m '+month+' -d '+day+' --dir '+'./'+' -g --area '+area)
		os.system(							'python get_era5_cds.py --2da -y '+year+' -m '+month+' -d '+day+' --dir '+'./'+' -g --area '+area)

		# Convert to HYSPILT compatible plot
		# In practice you may want to run the following in a separate script, after you have confirmed that all the data downloaded properly.
		d_out=								d_met
		month_namelist=						['-9999', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
		month_name=							month_namelist[int(month)]
		day_name=							str(int(day)).zfill(2)
		os.system(							'mv new_era52arl.cfg era52arl.cfg')				# use the cfg file created for the conversion.
		os.system(							'echo '+d_era52arl+'era52arl -i'+'./'+'ERA5_'+year+'.'+month_name+day_name+'.3dpl.grib -a'+'./'+'ERA5_'+year+'.'+month_name+day_name+'.2dpl.all.grib')
		os.system(							d_era52arl+'era52arl -i'+'./'+'ERA5_'+year+'.'+month_name+day_name+'.3dpl.grib -a'+'./'+'ERA5_'+year+'.'+month_name+day_name+'.2dpl.all.grib')
		os.system(							'mv DATA.ARL ./ERA5_'+year+str(int(month)).zfill(2)+str(int(day)).zfill(2)+'_global.ARL') 			# Rename immedinately
		os.system(							'mv ./ERA5_'+year+str(int(month)).zfill(2)+str(int(day)).zfill(2)+'_global.ARL '+d_met+'.')			# then move file
		os.system(							'rm '+'ERA5_'+year+'.'+month_name+day_name+'.3dpl.grib')
		os.system(							'rm '+'ERA5_'+year+'.'+month_name+day_name+'.2dpl.all.grib')


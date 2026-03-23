## This code runs within csf.prcs.py
import os, sys, glob, IPython, xarray as xr, numpy as np, pandas as pd, geopandas as gpd, pyproj, math, shutil
import matplotlib as mpl, matplotlib.pyplot as plt, contextily as ctx, matplotlib.cm as cmx
from mpl_toolkits.axes_grid1 import make_axes_locatable
sys.path.insert(0, "..")
import CT, FN

def _plot_csf(CFG, target, data, data_sampled, data_csf, tdump, sate, flux_unit, o_plot_wddiff, dfout):
	mpl.rcParams.update(					mpl.rcParamsDefault)
	mpl.rcParams['font.size']=				8
	figsize=								(7.24*1.0, 7.24*0.65)
	ncols_nrows=							(11,7)
	gs_kw=									dict(width_ratios=[1]*ncols_nrows[0], height_ratios=[1]*ncols_nrows[1])			# [1, 1] for equal ratio for two panel
	ss=										[0.00, 0.110, 0.963, 0.99, 0.00, 0.00]
	fig, ax=								plt.subplots(ncols=ncols_nrows[0], nrows=ncols_nrows[1], figsize=figsize, gridspec_kw=gs_kw)
	plt.subplots_adjust(					left=ss[0], bottom=ss[1], right=ss[2], top=ss[3], wspace=ss[4], hspace=ss[5])
	for i in range(0, ncols_nrows[0]):
		for j in range(0, ncols_nrows[1]):
			ax[j,i].axis('off')

	gs=										ax[0, 0].get_gridspec()
	axbig=									fig.add_subplot(gs[0:4, 0:3])
	axbig2=									fig.add_subplot(gs[5:8, 1:11])

	## 1. Map
	### City boundary
	dx=										0.98
	dy=										1.20
	axbig.set_xlim(							[ target['lon']-dx, target['lon']+dx] )
	axbig.set_ylim(							[ target['lat']-dy, target['lat']+dy] )
	axbig.tick_params(						axis='both', labelsize=6)
	axbig.tick_params(						axis='x', which='both', left=False, right=False, bottom=False, top=False, labelleft=False, labelbottom=False)
	axbig.tick_params(						axis='y', which='both', left=False, right=False, bottom=False, top=False, labelleft=False, labelbottom=False)
	ctx.add_basemap(						ax=axbig, alpha=1.0, crs='EPSG:4326', attribution=False, source=ctx.providers.OpenStreetMap.Mapnik)
	p_pp=									axbig.scatter(target['lon'], target['lat'], marker='o', color='black')
	cmap= cm=								plt.get_cmap('viridis')

	### Add pixel polygon column to gdf
	divider=								make_axes_locatable(axbig)
	cax=									divider.append_axes("bottom", size="2%", pad=0.03)	 # divide a subplot to put a figure and a color bar separately
	if CFG['SATE_INFO'][0] == 'oco3':
		voi=								'xco2'
		data=								data.loc[data['xco2_quality_flag'] == 0]
		vmin=								data[voi].sort_values().reset_index(drop=True)[int(len(data)*0.01)]
		vmax=								data[voi].sort_values().reset_index(drop=True)[int(len(data)*0.95)]
		alpha=								0.8
		txt=								r'OCO-3 SAM XCO$\mathregular{_{2}}$ [ppm]'
		alpha=								1.0
		cmap=								'viridis'
		p_sate=								data.plot(	ax=axbig, cax=cax, column=voi, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax,
															legend=True, legend_kwds={'label': txt, 'orientation': "horizontal"})
		clr_ortho=							'black'
	if CFG['SATE_INFO'][0] == 'trop':
		voi=								'no2'
		data=								data.loc[data['qa_value'] > 0.5]
		vmin=								data[voi].sort_values().reset_index(drop=True)[int(len(data)*0.00)]		# 0.
		vmax=								data[voi].sort_values().reset_index(drop=True)[int(len(data)*0.99)]		# 160.
		txt=								r'TROPOMI NO$\mathregular{_{2}}$ [$\mu$mol m$\mathregular{^{-2}}$]'
		alpha=								1.0
		cmap=								'viridis'
		p_sate=								data.plot(	ax=axbig, cax=cax, column=voi, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax,
															legend=True, legend_kwds={'label': txt, 'orientation': "horizontal"})
		clr_ortho=							'white'


	### Wind direction difference
	if o_plot_wddiff == True:
		p_wd_a3=							axbig.plot(		data_csf.iloc[0:-1]['lon_a3'], data_csf.iloc[0:-1]['lat_a3'], color='red',  lw=0.7)
		p_wd_a3=							axbig.scatter(	data_csf.iloc[0:-1]['lon_a3'], data_csf.iloc[0:-1]['lat_a3'], edgecolor='red', marker='o', s=3.0, lw=0.7, facecolor='none')
		for i in range(0, len(data_csf)):
			p_plume_width=					axbig.plot( data_csf.loc[i,['lon_a3_i','lon_a3_f']].values, data_csf.loc[i,['lat_a3_i','lat_a3_f']].values, color='magenta', lw=0.5)
			if i % 2 == 0:
				p_tdumpid=					axbig.text( data_csf.loc[i,'lon_a3'], data_csf.loc[i,'lat_a3'], str(data_csf.loc[i,'tdump_id']) , fontsize=4)
				p_wd=						axbig.text( data_csf.loc[i,'lon_a3'], data_csf.loc[i,'lat_a3'], str(int(data_csf.loc[i,'wd_diff_avg'])) , fontsize=4)

	### Traj
	p_traj=									axbig.plot(tdump['longitude'].values, tdump['latitude'].values, color='white', linewidth=0.8, linestyle='solid', marker='o', markersize=0.1)
	for i in range(0, len(data_sampled['tdump_id'].unique())):
		lon_ortho=							data_sampled.loc[data_sampled['tdump_id']==data_sampled['tdump_id'].unique()[i], 'lon_ortho'].values
		lat_ortho=							data_sampled.loc[data_sampled['tdump_id']==data_sampled['tdump_id'].unique()[i], 'lat_ortho'].values
		p_ortho_line=						axbig.plot(lon_ortho, lat_ortho, linestyle='dashed', color=clr_ortho, linewidth=0.40)
		lon_a3=								data_csf.loc[data_csf['tdump_id']==data_sampled['tdump_id'].unique()[i], 'lon_a3'].values[0]
		lat_a3=								data_csf.loc[data_csf['tdump_id']==data_sampled['tdump_id'].unique()[i], 'lat_a3'].values[0]
		a3=									data_csf.loc[data_csf['tdump_id']==data_sampled['tdump_id'].unique()[i], 'a3'].values[0]
		a4=									data_csf.loc[data_csf['tdump_id']==data_sampled['tdump_id'].unique()[i], 'a4'].values[0]

		tmp1=								data_sampled.loc[(data_sampled['tdump_id']==data_sampled['tdump_id'].unique()[i]) & (data_sampled['dist_ortho'] >= a3 - (a4/2.)) & (data_sampled['dist_ortho'] <= a3 + (a4/2.)) ].reset_index(drop=True)
		p_fmwh=								axbig.plot(tmp1['lon_ortho'], tmp1['lat_ortho'], linestyle='solid', color='red', linewidth=0.40)
		p_wd_a3=							axbig.scatter(	lon_a3, lat_a3, edgecolor='red', marker='o', s=3.0, lw=0.7, facecolor='none')

	### TXT
	if CFG['SATE_INFO'][0] == 'trop':
		axbig.text(0.04, 0.96, 'TROPOMI: '+pd.to_datetime(sate['time_utc']).strftime('%Y-%m-%d %H:%M')+' (UTC)', va='center', transform=axbig.transAxes, color='white', fontsize=7)

	if CFG['SATE_INFO'][0] == 'oco3':
		axbig.text(0.04, 0.96, 'OCO-3: '+pd.to_datetime(sate['time_utc']).strftime('%Y-%m-%d %H:%M')+' (UTC)', va='center', transform=axbig.transAxes, color='black', fontsize=7)



	## 2. Distance vs. voi plots
	### Set xlim, ylim
	xlim=									[-110., 110.]
	ylim=									[9999., -9999.]
	for i in range(0, len(data_csf)):
		if data_sampled.loc[data_sampled['tdump_id']==data_csf.loc[i,'tdump_id'], voi].min() < ylim[0]:
			ylim[0]=						data_sampled.loc[data_sampled['tdump_id']==data_csf.loc[i,'tdump_id'], voi].min()
		if data_sampled.loc[data_sampled['tdump_id']==data_csf.loc[i,'tdump_id'], voi].max() > ylim[1]:
			ylim[1]=						data_sampled.loc[data_sampled['tdump_id']==data_csf.loc[i,'tdump_id'], voi].max()

	if CFG['SATE_INFO'][0] == 'oco3':
		ylim=								[ylim[0]-0.05, ylim[1]+0.05]
		flux_varname=						'flux_co2'
	if CFG['SATE_INFO'][0] == 'trop':
		ylim=								[0., ylim[1]*1.10]
		flux_varname=						'flux_no2'

	pp_list=							[	(0,3), (0,4), (0,5), (0,6), (0,7), (0,8), (0,9), (0,10),  \
											(1,3), (1,4), (1,5), (1,6), (1,7), (1,8), (1,9), (1,10), \
											(2,3), (2,4), (2,5), (2,6), (2,7), (2,8), (2,9), (2,10), \
											(3,3), (3,4), (3,5), (3,6), (3,7), (3,8), (3,9), (3,10), ]

	norm=									mpl.colors.Normalize(vmin=0, vmax=len(data_sampled['tdump_id'].unique()))
	cmap= cm=								plt.get_cmap('rainbow')
	cnorm=									mpl.colors.Normalize(vmin=0., vmax=len(data_sampled['tdump_id'].unique()))
	scalarmap=								cmx.ScalarMappable(norm=cnorm, cmap=cmap)

	for i in range(0, len(data_csf)):
		pp=									pp_list[i]
		ax[pp].axis('on')
		if pp == (3,10):
			ax[pp].tick_params(				axis='x', which='both', left=False, right=False, bottom=True, top=False, labelleft=False, labelright=False, labelbottom=True, direction='in')
			ax[pp].tick_params(				axis='y', which='both', left=False, right=True, bottom=False, top=False, labelleft=False, labelright=True, labelbottom=False, direction='in')
			ax[pp].set_xlabel(				'Distance [km]')
		else:
			ax[pp].tick_params(				axis='x', which='both', left=False, right=False, bottom=True, top=False, labelleft=False, labelright=False, labelbottom=False, direction='in')
			ax[pp].tick_params(				axis='y', which='both', left=False, right=True, bottom=False, top=False, labelleft=False, labelright=False, labelbottom=False, direction='in')

		ax[pp].set_xlim(					xlim)
		ax[pp].set_ylim(					ylim)


		### Sampled soundings
		tmp=								data_sampled.loc[data_sampled['tdump_id']==data_csf.loc[i,'tdump_id']].reset_index(drop=True)
		p_var=								ax[pp].scatter(tmp['dist_ortho'], tmp[voi], marker='o', edgecolor='grey', facecolor='none', s=2., lw=0.4)

		### Gasussian curve
		clr=								'grey'
		if 'QF_gauss_abs' in data_csf.columns:
			if data_csf.loc[i,'QF_gauss_abs'] == 0:
				clr=							'limegreen'
			if data_csf.loc[i,'QF_gauss_abs'] == 1:
				clr=							'grey'
		x1=									data_csf.loc[i,'a3'] - (data_csf.loc[i,'a4']/2.)
		x2=									data_csf.loc[i,'a3'] + (data_csf.loc[i,'a4']/2.)
		y1=									FN._gaussian(x1, *data_csf.loc[i,['a0','a1','a2','a3','a4']].values)
		y2=									FN._gaussian(x2, *data_csf.loc[i,['a0','a1','a2','a3','a4']].values)
		p_fmwh=								ax[pp].scatter(data_csf.loc[i,'a3'], (y1+y2)/2., marker='o', color='red', s=5.)
		p_fmwh=								ax[pp].plot([x1, x2], [y1, y2], color='red', linewidth=0.70)
		varfit=								FN._gaussian(np.arange(xlim[0],xlim[1]), *data_csf.loc[i,['a0','a1','a2','a3','a4']].values)
		p_varfit=							ax[pp].plot(np.arange(xlim[0],xlim[1]), varfit, color=clr, linewidth=1.0)

		### Text, error metrics
		fontsize=							4.0
		if (np.isnan(data_csf.loc[i,flux_varname]) == False):
			p_flux=							ax[pp].text(0.85, 1.00-(0.105*1), str(data_csf.loc[i,'tdump_id']+1),						transform=ax[pp].transAxes, fontsize=fontsize, color='black', horizontalalignment='left')
			p_flux=							ax[pp].text(0.03, 1.00-(0.105*1), r'Flux= '+ str(int(np.round(data_csf.loc[i,flux_varname]))), transform=ax[pp].transAxes, fontsize=fontsize, color='black', horizontalalignment='left')
			p_flux=							ax[pp].text(0.03, 1.00-(0.105*2), r'$\sigma$$\mathregular{_{center}}$= '+ str(int(np.round(data_csf.loc[i,'d_center'],0))), transform=ax[pp].transAxes, fontsize=fontsize, color='black', horizontalalignment='left')
			p_flux=							ax[pp].text(0.03, 1.00-(0.105*3), r'$\sigma$$\mathregular{_{left}}$= '+ str(int(np.round(data_csf.loc[i,'d_left'],0))), transform=ax[pp].transAxes, fontsize=fontsize, color='black', horizontalalignment='left')
			p_flux=							ax[pp].text(0.03, 1.00-(0.105*4), r'$\sigma$$\mathregular{_{right}}$= '+ str(int(np.round(data_csf.loc[i,'d_right'],0))), transform=ax[pp].transAxes, fontsize=fontsize, color='black', horizontalalignment='left')
			p_flux=							ax[pp].text(0.03, 1.00-(0.105*5), r'$\sigma$$\mathregular{_{gap}}$= '+ str(int(np.round(data_csf.loc[i,'d_gap'],0))), transform=ax[pp].transAxes, fontsize=fontsize, color='black', horizontalalignment='left')
			p_flux=							ax[pp].text(0.03, 1.00-(0.105*6), r'r$\mathregular{^{2}}$= '+str(np.round(data_csf.loc[i,'rsq_detrend'],2)),	transform=ax[pp].transAxes, fontsize=fontsize, color='black', horizontalalignment='left')
			a2tmp=							np.round(data_csf.loc[i,'a2sig']/(data_csf.loc[i,'a2']+1.E-9)*100.)
			a4tmp=							np.round(data_csf.loc[i,'a4sig']/(data_csf.loc[i,'a4']+1.E-9)*100.)
			if a2tmp > 100.:
				a2tmp=						'> 100'
			else:
				a2tmp=						str(a2tmp)
			if a4tmp > 100.:
				a4tmp=						'> 100'
			else:
				a4tmp=						str(a4tmp)
			p_flux=							ax[pp].text(0.03, 1.00-(0.105*7), r'$\sigma$$\mathregular{_{a2}}$= '+ a2tmp, transform=ax[pp].transAxes, fontsize=fontsize, color='black', horizontalalignment='left')
			p_flux=							ax[pp].text(0.03, 1.00-(0.105*8), r'$\sigma$$\mathregular{_{a4}}$= '+ a4tmp, transform=ax[pp].transAxes, fontsize=fontsize, color='black', horizontalalignment='left')
			p_flux=							ax[pp].text(0.03, 1.00-(0.105*9), r'MAPE= '+ str(np.round(data_csf.loc[i,'mape'],4)), transform=ax[pp].transAxes, fontsize=fontsize, color='black', horizontalalignment='left')



	## 3. NOx cross sectional flux vs. time
	p_xlabel=								axbig2.set_xlabel('Transport time [hours]')
	p_ylabel=								axbig2.set_ylabel('Cross-sectional flux '+flux_unit)

	if CFG['SATE_INFO'][0] == 'oco3':
		if 'QF_gauss_abs' in data_csf.columns:
			p_co2flux=						axbig2.scatter(data_csf.loc[data_csf['QF_gauss_abs']==0,'age_hours'], data_csf.loc[data_csf['QF_gauss_abs']==0,'flux_co2'], marker='o', color='dodgerblue')
			p_co2flux=						axbig2.scatter(data_csf.loc[data_csf['QF_gauss_abs']==1,'age_hours'], data_csf.loc[data_csf['QF_gauss_abs']==1,'flux_co2'], marker='x', color='dodgerblue')
		else:
			p_co2flux=						axbig2.scatter(data_csf['age_hours'], data_csf['flux_co2'], marker='x', color='dodgerblue')


	if CFG['SATE_INFO'][0] == 'trop':
		if 'QF_gauss_abs' in data_csf.columns:
			p_no2flux=						axbig2.scatter(data_csf.loc[data_csf['QF_gauss_abs']==0,'age_hours'], data_csf.loc[data_csf['QF_gauss_abs']==0,'flux_no2'], marker='o', color='dodgerblue')
			p_no2flux=						axbig2.scatter(data_csf.loc[data_csf['QF_gauss_abs']==1,'age_hours'], data_csf.loc[data_csf['QF_gauss_abs']==1,'flux_no2'], marker='x', color='dodgerblue')
		else:
			p_no2flux=						axbig2.scatter(data_csf['age_hours'], data_csf['flux_no2'], marker='x', color='dodgerblue')
		if CFG['o_calc_nox'] == True:
			if 'QF_gauss_abs' in data_csf.columns:
				p_noxflux=						axbig2.scatter(data_csf.loc[data_csf['QF_gauss_abs']==0,'age_hours'], data_csf.loc[data_csf['QF_gauss_abs']==0,'flux_nox'], marker='o', color='orangered')
				p_noxflux=						axbig2.scatter(data_csf.loc[data_csf['QF_gauss_abs']==1,'age_hours'], data_csf.loc[data_csf['QF_gauss_abs']==1,'flux_nox'], marker='x', color='orangered')
			else:
				p_noxflux=						axbig2.scatter(data_csf['age_hours'], data_csf['flux_nox'], marker='x', color='orangered')

			x=								np.arange(0., 6., 0.1)
			popt_lifetime=					(data_csf['emis_nox'].values[0], data_csf['t_pss'].values[0])
			y=								FN._nox_lifetime(x, *popt_lifetime)
			p_fit=							axbig2.plot(x, y, color='orangered')
			p_text=							axbig2.text(0.5, 0.85, 'NOx Emission rate= '+str(np.round(data_csf.loc[0,'emis_nox'],1))+' '+flux_unit+' ('+str(np.round(data_csf.loc[0,'emis_nox']/1.E3*24.*365.,1))+' kt/yr)', transform=axbig2.transAxes)
			p_text=							axbig2.text(0.5, 0.75, 'NOx Lifetime= '+str(np.round(data_csf.loc[0,'nox_lifetime'],1))+' hrs', transform=axbig2.transAxes)

			if data_csf.loc[0,'t_pss'] != -9999.:
				p_pss=						axbig2.axvline(x=data_csf.loc[0,'t_pss'], color='grey', linestyle='dotted')
		#p_text=								axbig2.text(0.5, 0.65, 'wso= '+str(np.round(data_csf['wso'].mean(),1))+' m/s', transform=axbig2.transAxes)

	## SAVE
	fig.savefig(dfout+'.png', dpi=300)
	plt.close('all')
	plt.cla()

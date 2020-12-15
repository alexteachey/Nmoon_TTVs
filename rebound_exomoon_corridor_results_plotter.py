from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas
import socket
import os
import time
import corner 
import traceback


#### THIS CODE WILL PRODUCE THE PLOTS YOU WANT TO GENERATE FOR THE MULTI-MOON EXOCORRIDOR PAPER
###### NOT TO BE CONFUSED WITH REBOUND_RESULTS_INTERPRETTER.PY, WHICH BECAME MOSTLY A MACHINE LEARNING PLAYGROUND.
######## THIS IS JUST ANALYSIS!!!



if socket.gethostname() == 'tethys.asiaa.sinica.edu.tw':
	#projectdir = '/data/tethys/Documents/Projects/NMoon_TTVs'
	projectdir = '/run/media/amteachey/Auddy_Akiti/Teachey/Nmoon_TTVs'
elif socket.gethostname() == 'Alexs-MacBook-Pro.local':
	projectdir = '/Users/hal9000/Documents/Projects/Nmoon_TTVsim'
else:
	projectdir = input('Please input the project directory: ')

positionsdir = projectdir+'/sim_positions'
ttvfiledir = projectdir+'/sim_TTVs'
LSdir = projectdir+'/sim_periodograms'
modeldictdir = projectdir+'/sim_model_settings'
plotdir = projectdir+'/sim_plots'


#### FIRST, MAKE A CORNER PLOT from projectdir+'/simulation_summary.csv'
try:
	sim_summary = pandas.read_csv(projectdir+'/simulation_summary.csv')
	simsum_cols = sim_summary.columns
	simsum_dict = {}

	for ncol, col in enumerate(simsum_cols):
		if ncol == 0:
			corner_data = np.zeros(shape=(len(np.array(sim_summary[col])), len(simsum_cols)-1))
		simsum_dict[col] = np.array(sim_summary[col])
		if ncol != 0:
			corner_data.T[ncol-1] = np.array(sim_summary[col]) ### skip the sim number column.

	isfinites = np.isfinite(corner_data.T[-1])

	### conver the NaNs to a randomly generated number (so as not to skew)
	corner_data.T[-1] = np.nan_to_num(corner_data.T[-1])	

	#### plot the corner plot!
	figure = corner.corner(corner_data[np.isfinite(corner_data)], labels=simsum_cols[1:], quantiles=[0.16, 0.5, 0.84], showtitles=True, title_kwargs={"fontsize":12})
	plt.show()


	#### plot each histogram individually
	for col in simsum_cols:
		if col == 'TTV_rmsamp_sec':
			plt.xlabel('TTV RMS amplitude [minutes]')
			n, bins, edges = plt.hist(simsum_dict[col]/60, bins=20, facecolor='DodgerBlue', edgecolor='k', alpha=0.7)	
		
		elif col == 'Mmoons_over_Mplan':
			n, bins, edges = plt.hist(simsum_dict[col], bins=20, facecolor='DodgerBlue', edgecolor='k', alpha=0.7)
			plt.xlabel(r'$\sum M_S \, / \, M_P$')

		else:
			n, bins, edges = plt.hist(simsum_dict[col], bins=20, facecolor='DodgerBlue', edgecolor='k', alpha=0.7)
			plt.xlabel(col)
		plt.show()



	#### PLOT MEGNO VERSUS SPOCK STABILITY!
	nspock_systems = 0
	for moonnum in np.arange(3,6,1):
		nmoon_idxs = np.where(simsum_dict['nmoons'] == moonnum)[0]
		nspock_systems += len(nmoon_idxs)
		plt.scatter(simsum_dict['SPOCK_survprop'][nmoon_idxs], simsum_dict['MEGNO'][nmoon_idxs], s=20, alpha=0.5, label=str(moonnum))

	plt.legend()
	plt.xlabel('SPOCK survival probability')
	plt.ylabel('MEGNO')
	plt.show()

	##### GRAB ALL THE VALUES WHERE SPOCK PROBABILITY OF SURVIVABILITY IS >= 90 percent.
	####### OF THAT SUBSET, LOOK AT THE DISTRIBUTION OF MEGNO VALUES.
	######### LET THE UPPER LIMIT MEGNO FOR N = 1 and N = 2 moons be within the 1-sigma contour.

	spock_above90_idxs = np.where(simsum_dict['SPOCK_survprop'] >= 0.9)[0]
	megnos_for_spock_above90 = simsum_dict['MEGNO'][spock_above90_idxs]
	median_megno, megno_std = np.nanmedian(megnos_for_spock_above90), np.nanstd(megnos_for_spock_above90)
	megno_16pct, megno_84pct = np.nanpercentile(megnos_for_spock_above90, 16), np.nanpercentile(megnos_for_spock_above90, 84) ### 1 sigma boundary
	megno_2point5pct, megno_97point5pct = np.nanpercentile(megnos_for_spock_above90, 2.5), np.nanpercentile(megnos_for_spock_above90, 97.5) ### 2 sigma boundary
	megno_twosig_upperlim = median_megno + (2*megno_std)

	print('# systems with SPOCK survival >= 90% = ', len(spock_above90_idxs))
	print('Percent total (3 or more moons) = ', (len(spock_above90_idxs) / nspock_systems)*100)
	print('median MEGNO (SPOCK SURVIVAL >= 90%): ', median_megno)
	print('sigma = ', megno_std)
	#print('two-sig upper limit = ', megno_twosig_upperlim)
	print('megno_16pct, megno_84pct (1sig) = ', megno_16pct, megno_84pct)
	print('megno 2.5pct, megno 97.5 pct (2sig) = ', megno_2point5pct, megno_97point5pct)

	##### now find all the "GOOD MEGNO IDXs" that fit this bill
	good_megno_idxs = np.where(simsum_dict['MEGNO'] <= megno_twosig_upperlim)[0]







except:
	traceback.print_exc()
	raise Exception('something happened.')



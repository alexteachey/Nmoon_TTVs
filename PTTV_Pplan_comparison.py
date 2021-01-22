from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas
import time
import os
import matplotlib.cm as cm 
from scipy.stats import loguniform
import traceback

try:
	##### this script will compare the simulated PTTV-vs_Pplan plot with the mazeh catalog version.
	####### AND -- attempt to compute a moon probability (a la your paper) based on these abundances.

	projectdir = '/data/tethys/Documents/Projects/NMoon_TTVs'
	sim_heatmap = np.load(projectdir+'/simulated_PTTV10-1500_Pplan2-100_20x20_heatmap.npy', allow_pickle=True)[0]
	real_heatmap = np.load(projectdir+'/mazeh_PTTV10-1500_Pplan2-100_20x20_heatmap.npy', allow_pickle=True)[0]

	#### HEATMAPS ARE *NOT* MAPS OF P(PTTV | Pplan)! 
	#### for that you need each bin to be divided by the total number of systems in the Pplan column!
	#### That is: P(P_TTV | P_plan) = N(P_TTV) / N(Pplan) -- RIGHT?!?! (that's probability of being in those bins)


	#### DEFINITIVE ANSWER HERE ####
	"""
	sim_heatmap.T, origin="lower" is the correct orientation for plotting Pplan on the x-axis and P_TTV on the y-axis (see rebound_sim_plotter results).
	let's call the 2D array npp. npp[0] is the first COLUMN of Pplans. Thus, the lower-left box is npp[0][0]. The upper-left box is npp[0][-1].
	column_sums = []
	for i in np.arange(0,19,1):
		colsum = npp[i]
		column_sums.append(colsum)
	column_sums = np.array(column_sums)

	#### column_sums is the equivalent to np.nansum(npp, axis=1)
	assert np.all(column_sums == np.nansum(npp, axis=1))

	SOOOO YOU WANT TO DIVIDE EVERY BIN BY THE SUM OF IT'S COLUMN TO GET P(P_TTV | P_Plan)

	"""

	sim_column_sums = []
	for i in np.arange(0,19,1):
		colsum = np.nansum(sim_heatmap[i])
		sim_column_sums.append(colsum)
	sim_column_sums = np.array(sim_column_sums)
	#### column_sums is the equivalent to np.nansum(npp, axis=1)
	assert np.all(sim_column_sums == np.nansum(sim_heatmap, axis=1))
	### COLUMN SUMS ARE THE NUMBER OF SYSTEMS IN EACH PERIOD BIN.

	real_column_sums = []
	for i in np.arange(0,19,1):
		colsum = np.nansum(real_heatmap[i])
		real_column_sums.append(colsum)
	real_column_sums = np.array(real_column_sums)
	#### column_sums is the equivalent to np.nansum(npp, axis=1)
	assert np.all(real_column_sums == np.nansum(real_heatmap, axis=1))
	### COLUMN SUMS ARE THE NUMBER OF SYSTEMS IN EACH PERIOD BIN.	

	"""
	NOTE: you need to TRANSPOSE the matrix so that x is along the x-axis and y- is along the y-axis.
	See documentation note here: https://numpy.org/doc/stable/reference/generated/numpy.histogram2d.html
	Please note that the histogram does not follow the Cartesian convention where x values are on the abscissa and y values on the ordinate axis. Rather, x is histogrammed along the first dimension of the array (vertical), and y along the second dimension of the array (horizontal). This ensures compatibility with histogramdd.
	"""


	#### BELOW ARE HEATMAPS! THE NUMBER IN EACH BIN!
	#### SOMETHING IS WEIRD ABOUT THE GKDE -- try a heatmap (hist2d)
	xbins = np.logspace(np.log10(10), np.log10(1500), 20) #### planet periods
	ybins = np.logspace(np.log10(2), np.log10(100), 20) #### P_TTVs

	xcenters, ycenters, xwidths, ywidths = [], [], [], [] #### Pplan_centers, P_TTV centers.
	for nx,x in enumerate(xbins):
		#try:
		if nx != 19:
			xcenters.append(np.nanmean((xbins[nx+1], xbins[nx])))
			xwidths.append(xbins[nx+1] - xbins[nx])
		#except:
		#	traceback.print_exc()
		#	pass
	xcenters = np.array(xcenters)
	xwidths = np.array(xwidths)
	Pplan_centers, Pplan_widths = xcenters, xwidths

	for ny,y in enumerate(ybins):
		#try:
		if ny != 19:
			ycenters.append(np.nanmean((ybins[ny+1], xbins[ny])))
			ywidths.append(ybins[ny+1] - ybins[ny])
		#except:
		#	traceback.print_exc()
		#	pass
	ycenters = np.array(ycenters)
	ywidths = np.array(ywidths)
	Pttv_centers, Pttv_widths = ycenters, ywidths 



	fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True, figsize=(6,10))
	ax1.imshow(sim_heatmap.T, origin='lower', cmap=cm.coolwarm)
	ax1.set_yticks(ticks=np.arange(0,len(ybins),5))
	ax1.set_yticklabels(labels=np.around(np.log10(ybins[::5]),2))
	ax1.set_ylabel(r'$\log_{10} \, P_{\mathrm{TTV}}$ [epochs] (sim)')
	ax1.set_title('Simulated / Real Systems Distribution')

	ax2.imshow(real_heatmap.T, origin='lower', cmap=cm.coolwarm)
	ax2.set_xticks(ticks=np.arange(0,len(xbins),5))
	ax2.set_xticklabels(labels=np.around(np.log10(xbins[::5]),2))
	ax2.set_yticks(ticks=np.arange(0,len(ybins),5))
	ax2.set_yticklabels(labels=np.around(np.log10(ybins[::5]), 2))
	ax2.set_ylabel(r'$\log_{10} \, P_{\mathrm{TTV}}$ [epochs] (real)')
	ax2.set_xlabel(r'$\log_{10} \, P_{\mathrm{P}}$ [days]')
	plt.tight_layout()

	plt.show()


	### MAKE PROBABILITY MAPS P(P_TTV | P_Pplan)
	fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True, figsize=(6,10))
	ax1.imshow(sim_heatmap.T / sim_column_sums, origin='lower', cmap=cm.coolwarm)
	ax1.set_yticks(ticks=np.arange(0,len(ybins),5))
	ax1.set_yticklabels(labels=np.around(np.log10(ybins[::5]),2))
	ax1.set_ylabel(r'$\log_{10} \, P_{\mathrm{TTV}}$ [epochs] (sim)')
	ax1.set_title('Simulated / Real Systems P(P_TTV | P_Pplan)')

	ax2.imshow(real_heatmap.T / real_column_sums, origin='lower', cmap=cm.coolwarm)
	ax2.set_xticks(ticks=np.arange(0,len(xbins),5))
	ax2.set_xticklabels(labels=np.around(np.log10(xbins[::5]),2))
	ax2.set_yticks(ticks=np.arange(0,len(ybins),5))
	ax2.set_yticklabels(labels=np.around(np.log10(ybins[::5]), 2))
	ax2.set_ylabel(r'$\log_{10} \, P_{\mathrm{TTV}}$ [epochs] (real)')
	ax2.set_xlabel(r'$\log_{10} \, P_{\mathrm{P}}$ [days]')
	plt.tight_layout()

	plt.show()


	sim_prob_Pttv_given_Pplan = sim_heatmap.T / sim_column_sums #### READY TO PLOT AS IS, WITH ORIGIN="LOWER"
	real_prob_Pttv_given_Pplan = real_heatmap.T / real_column_sums  ### READ TO PLOT AS IS, WITH ORIGIN="LOWER"

	loguniform_prob_multiplier = np.linspace(1/19,1/19,19) #### uniform in logbins
	print('np.nansum(loguniform_prob_multiplier = ', np.nansum(loguniform_prob_multiplier))
	xbin_range = np.nanmax(xbins) - np.nanmin(xbins)
	propto_Pplan_areas = xcenters * xwidths
	propto_Pplan_total_area = np.nansum(propto_Pplan_areas)
	propto_Pplan_prob_multiplier = propto_Pplan_areas / propto_Pplan_total_area 
	print('np.nansum(propto_Pplan_prob_multiplier = ', np.nansum(propto_Pplan_prob_multiplier))


	#### generate numerators!
	loguniform_numerator = sim_prob_Pttv_given_Pplan * loguniform_prob_multiplier
	propto_Pplan_numerator = sim_prob_Pttv_given_Pplan * propto_Pplan_prob_multiplier


	### MAKE PROBABILITY MAPS P(P_TTV | P_Pplan)
	fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True, figsize=(6,10))
	ax1.imshow(loguniform_numerator, origin='lower', cmap=cm.coolwarm)
	ax1.set_yticks(ticks=np.arange(0,len(ybins),5))
	ax1.set_yticklabels(labels=np.around(np.log10(ybins[::5]),2))
	#ax1.set_ylabel(r'$\log_{10} \, P_{\mathrm{TTV}}$ [epochs] (log-uniform)')
	#ax1.set_title(r'$P(P_{\mathrm{TTV}} \, | \, moon, \, P_{\mathrm{P)} \, P(moon \, | \, P_{\mathrm{P}})$')

	ax2.imshow(propto_Pplan_numerator, origin='lower', cmap=cm.coolwarm)
	ax2.set_xticks(ticks=np.arange(0,len(xbins),5))
	ax2.set_xticklabels(labels=np.around(np.log10(xbins[::5]),2))
	ax2.set_yticks(ticks=np.arange(0,len(ybins),5))
	ax2.set_yticklabels(labels=np.around(np.log10(ybins[::5]), 2))
	#ax2.set_ylabel(r'$\log_{10} \, P_{\mathrm{TTV}}$ [epochs] $\propto P_{\mathrm{P}}$')
	#ax2.set_xlabel(r'$\log_{10} \, P_{\mathrm{P}}$ [days]')
	#plt.tight_layout()

	plt.show()


	#### NOW PUT IT ALL TOGETHER (NUMERATOR AND DENOMINATOR)
	Pmoon_given_Pttv_and_Pplan_loguniform = loguniform_numerator / real_heatmap.T
	Pmoon_given_Pttv_and_Pplan_propto_pplan = propto_Pplan_numerator / real_heatmap.T 

	### MAKE PROBABILITY MAPS P(P_TTV | P_Pplan)
	fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True, figsize=(6,10))
	ax1.imshow(Pmoon_given_Pttv_and_Pplan_loguniform, origin='lower', cmap=cm.coolwarm)
	ax1.set_yticks(ticks=np.arange(0,len(ybins),5))
	ax1.set_yticklabels(labels=np.around(np.log10(ybins[::5]),2))
	#ax1.set_ylabel(r'$\log_{10} \, P_{\mathrm{TTV}}$ [epochs] (log-uniform)')
	#ax1.set_title(r'$P(P_{\mathrm{TTV}} \, | \, moon, \, P_{\mathrm{P)} \, P(moon \, | \, P_{\mathrm{P}})$')

	ax2.imshow(Pmoon_given_Pttv_and_Pplan_propto_pplan, origin='lower', cmap=cm.coolwarm)
	ax2.set_xticks(ticks=np.arange(0,len(xbins),5))
	ax2.set_xticklabels(labels=np.around(np.log10(xbins[::5]),2))
	ax2.set_yticks(ticks=np.arange(0,len(ybins),5))
	ax2.set_yticklabels(labels=np.around(np.log10(ybins[::5]), 2))
	#ax2.set_ylabel(r'$\log_{10} \, P_{\mathrm{TTV}}$ [epochs] $\propto P_{\mathrm{P}}$')
	#ax2.set_xlabel(r'$\log_{10} \, P_{\mathrm{P}}$ [days]')
	#plt.tight_layout()

	plt.show()	







	raise Exception('this is far as you want to go right now')


	#### NOW WE WANT TO COMPUTE P(moon | PTTV, PPlan). That is given by
	"""
	P(moon | PTTV, Pplan) = (P(PTTV | moon, Pplan) * P(moon | Pplan)) / P(PTTV | Pplan)
	we just computed the denominator above (real_prob_Pttv_given_Pplan)

	First term in the numerator is also computed above: sim_prob_Pttv_given_Pplan. (sim has moons in it!)
	The second term in the numerator we will want to use some distribution of P(moon | Pplan). Uniform, loguniform, P, 1/P, P^2, 1/P^2, etc.
	
	THESE ARE CHANGING VALUES, SO LET LET Pplan in these distributions = P_center.

	
	"""








	#### DIVIDE THE sim_heatmap by the real_heatmap to see what P(moon | Pttv, Pp) looks like for LOGUNIFORM P(moon | Pp)
	"""
	logxbins = np.logspace(np.log10(10),np.log10(1500),20)
	logxwidths = []
	for nlogx,logx in enumerate(logxbins):
		try:
			logxwidths.append(logxbins[nlogx+1] - logxbins[nlogx])
		except:
			pass
	logxwidths = np.array(logxwidths)
	#### for UNIFORM AREAS IN LOG SPACE, YOU NEED HEIGHTS TO BE 1/logwidths
	logyheights = 1/logxwidths  ### at smaller values, widths are narrower, so heights are taller



	linxbins = np.linspace(10,1500,20)
	finxwidths = []
	for nlinx,linx in enumerate(linxbins):
		try:
			linxwidths.append(linxbins[nlinx+1] - linxbins[nlinx])
		except:
			pass
	linxwidths = np.array(linxwidths)
	#### for UNIFORM AREAS IN LINEAR SPACE, HEIGHTS NEED TO BE 1/linxwidths
	linyheights = 1/linxwidths 
	"""

	uniform_draws = np.random.uniform(10,1500,size=10000)
	loguniform_draws = loguniform.rvs(10,1500,size=10000)
	logbins = np.logspace(np.log10(10), np.log10(1500),20)

	fig, (ax1, ax2) = plt.subplots(2)
	ax1.hist(uniform_draws, bins=np.linspace(10,1500,20))
	ax1.set_ylabel('uniform draws, uniform binning')
	ax2.hist(uniform_draws, bins=logbins)
	ax2.set_ylabel('uniform draws, log binning')
	plt.show()


	uniform_nperbin = plt.hist(uniform_draws, bins=logbins)[0]
	plt.xscale('log')
	plt.title('uniform nperbin')
	plt.show()
	uniform_nperbin_20x20 = np.zeros(shape=(19,19)) #### there are actually 19 bins on a side.
	for i in np.arange(0,19,1):
		uniform_nperbin_20x20[i] = uniform_nperbin


	loguniform_nperbin = plt.hist(loguniform_draws, bins=logbins)[0]
	plt.title('loguniform nperbin')
	plt.xscale('log')
	plt.show()
	loguniform_nperbin_20x20 = np.zeros(shape=(19,19)) #### there are actually 19 bins on a side.
	for i in np.arange(0,19,1):
		loguniform_nperbin_20x20[i] = loguniform_nperbin

	uniform_numerator = sim_heatmap.T * uniform_nperbin_20x20
	loguniform_numerator = sim_heatmap.T * loguniform_nperbin_20x20 

	uniform_Pmoon = uniform_numerator / real_heatmap.T
	loguniform_Pmoon = loguniform_numerator / real_heatmap.T

	fig, (ax1,ax2) = plt.subplots(2, figsize=(6,10), sharex=True, sharey=True)

	ax1.imshow(uniform_Pmoon, origin='lower', cmap=cm.coolwarm)
	ax1.set_xticks(ticks=np.arange(0,len(xbins),5))
	ax1.set_xticklabels(labels=np.around(np.log10(xbins[::5]),2))
	ax1.set_yticks(ticks=np.arange(0,len(ybins),5))
	ax1.set_yticklabels(labels=np.around(np.log10(ybins[::5]), 2))
	#ax1.set_ylabel(r'$\log_{10} \, P_{\mathrm{TTV}}$ [epochs] (real)')
	#ax1.set_xlabel(r'$\log_{10} \, P_{\mathrm{P}}$ [days]')
	ax1.set_ylabel(r'$P(\mathrm{moon} \, | \, P_{\mathrm{TTV}}, \, P_{\mathrm{P}})$ (uniform)')
	#ax1.set_title('Uniform and Log(uniform) P(moon | P_TTV, P_Plan)')


	ax2.imshow(loguniform_Pmoon, origin='lower', cmap=cm.coolwarm)
	ax2.set_xticks(ticks=np.arange(0,len(xbins),5))
	ax2.set_xticklabels(labels=np.around(np.log10(xbins[::5]),2))
	ax2.set_yticks(ticks=np.arange(0,len(ybins),5))
	ax2.set_yticklabels(labels=np.around(np.log10(ybins[::5]), 2))
	#ax2.set_ylabel(r'$\log_{10} \, P_{\mathrm{TTV}}$ [epochs] (real)')
	ax2.set_ylabel(r'$P(\mathrm{moon} \, | \, P_{\mathrm{TTV}}, \, P_{\mathrm{P}})$ (loguniform)')
	ax2.set_xlabel(r'$\log_{10} \, P_{\mathrm{P}}$ [days]')


	plt.tight_layout()
	plt.show()









except:
	traceback.print_exc()
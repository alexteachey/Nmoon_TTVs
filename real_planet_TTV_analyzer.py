from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas
import os
import time
import traceback
from astropy.timeseries import LombScargle
from astropy.io import ascii
from scipy.optimize import curve_fit
import re
from scipy.stats import gaussian_kde 
import matplotlib.cm as cm 
from scipy.special import factorial
from moonpy import *

"""
#### THIS SCRIPT IS GOING TO PRODUCE ONE PLOT -- the plot of P_TTV vs P_plan. 
###### to do this, we will 

1) import the O-C values (Table3_O-C.csv),
2) run a periodogram
3) use the highest peak period as a period to fit a sinusoid
4) calculate a BIC and delta-BIC
5) grab the planet's orbital period (nasa exoplanet archive will be easiet)
6) make a scatter plot in P_TTV -- P_plan space.
7) compute a gaussian kernel density estimator
8) sav the GKDE as a pickle, so you can import it and use it elsewhere to calculate a probability for a given P_TTV and P_plan.


"""

def n_choose_k(n,k):
	numerator = factorial(n)
	denominator = factorial(k) * factorial(n - k)
	return numerator / denominator

"""
def binomial_probability(ntrials,nhits,phit):
	#### computes an expectation value curve (ish) for nhits, based on ntrials and hit probability.
	#### for example: if you have a coin with p(heads) = 0.5, and ntrials = 40, the curve maximum is at n=20.
	#### it falls off on either side of that. Thus, if you have nhits = 10, you can read off the probability of getting that value
	#### from the curve. (hint: it's low). 
	#### if you want to compute the UNKNOWN VALUE phit, you will want to test a range of phits (with given ntrials and nhits), 
	#### compute the curve, and find the phit for which the function you're calculating here is at maximum (~zero slope) 
	#### for n = nhits. That is, whatever you pulled out is the maximum probability value, and the uncertainty comes from that.
	#### you need to think about your application carefully before applying this.
"""

def chisquare(data, model, errors):
	return np.nansum(((data - model)**2) / errors**2)



try:

	show_plots = input('Do you want to show plots (for debugging)? y/n: ')

	OCfile = pandas.read_csv('/data/tethys/Documents/Software/MoonPy/Table3_O-C.csv')
	KOIs = np.array(OCfile['KOI']).astype(str)
	epochs = np.array(OCfile['n']).astype(int)
	OCmin = np.array(OCfile['O-C_min']).astype(str)
	OCmin_err = np.array(OCfile['O-C_err']).astype(str)


	### PURGE THOSE GODDAMN SPECIAL CHARACTERS OUT OF OCmin and OCmin_err
	OCmin_clean, OCmin_err_clean = [], []

	for OC, OCerr in zip(OCmin, OCmin_err):
		### THIS NASTY NESTED FOR LOOP IS BROUGHT TO YOU BY USING NON-NUMERIC CHARACTERS IN A NUMERIC COLUMN INSTEAD OF USING FLAGS.
		OCclean = ''
		for val in OC:
			if val in ['-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
				OCclean = OCclean+val
			else:
				pass
		OCmin_clean.append(float(OCclean))

		OCerrclean = ''	
		for val in OCerr:
			if val in ['-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
				OCerrclean = OCerrclean+val
			else:
				pass
		OCmin_err_clean.append(float(OCerrclean))

	OCmin = np.array(OCmin_clean)
	OCmin_err = np.array(OCmin_err_clean)


	unique_KOIs = np.unique(KOIs)





	cumkois = ascii.read('/data/tethys/Documents/Software/MoonPy/cumkois.txt')
	kepler_names = np.array(cumkois['kepler_name'])
	kepois = np.array(cumkois['kepoi_name'])
	kepoi_periods = np.array(cumkois['koi_period'])

	##### FIND THE HADDEN & LITHWICK KOIs.
	try:
		HL_KOIs = np.load('/data/tethys/Documents/Projects/NMoon_TTVs/Hadden_Lithwick_posteriors/HLKOIs.npy')
		print('loaded HL_KOIs...')
	
	except:
		HL_KOIs = []
		print('generating HL_KOIs...')
		HLplanet_list = np.load('/data/tethys/Documents/Projects/NMoon_TTVs/Hadden_Lithwick_posteriors/HLplanet_list.npy')
		for HLplanet in HLplanet_list:
			if HLplanet.startswith('Kepler'):
				#### find the kepoi in cumkois:
				HL_kepoi_idx = np.where(kepler_names == HLplanet[:-1]+' '+HLplanet[-1])[0]
				HL_kepoi = kepois[HL_kepoi_idx]
			else:
				HL_kepoi = HLplanet #### of the form K0001.01, etc

			HL_KOI = HL_kepoi
			while HL_KOI.startswith('K') or HL_KOI.startswith('0'):
				HL_KOI = HL_KOI[1:]
			HL_KOI = 'KOI-'+HL_KOI
			HL_KOIs.append(HL_KOI)
		HL_KOIs = np.array(HL_KOIs)
		assert len(HL_KOIs) == len(HLplanet_list)
		np.save('/data/tethys/Documents/Projects/NMoon_TTVs/Hadden_Lithwick_posteriors/HLKOIs.npy', HL_KOIs)


	kepoi_nums = []
	system_nums = []
	for kepoi in kepois:
		kepoi_num = kepoi
		while (kepoi_num.startswith('K')) or (kepoi_num.startswith('0')):
			kepoi_num = kepoi_num[1:]
		kepoi_nums.append(kepoi_num)
		system_nums.append(kepoi_num[:kepoi_num.find('.')]) #### leaves off the .01, .02, .03, etc.
	kepoi_nums = np.array(kepoi_nums)
	system_nums = np.array(system_nums)
	kepois = kepoi_nums

	#### determine if the kepois are in multi-planet systems
	kepoi_multi = []
	kepoi_multi_period_ratios = []

	for nkep,kep in enumerate(kepois): ### kepois are already strings
		kepoi_number = kep[:kep.find('.')] #### leaves off the final .01, .02, .03, etc.
		#### find how many entries match this in system_nums.
		all_system_planet_idxs = np.where(kepoi_number == system_nums)[0]
		all_system_planet_periods = kepoi_periods[all_system_planet_idxs]
		#

		nplanets_in_system = len(all_system_planet_idxs)
		if nplanets_in_system > 1:
			kepoi_multi.append(True)

		elif nplanets_in_system == 1:
			kepoi_multi.append(False)
		else:
			raise Exception('something weird happening with the single / multi counter.')
	kepoi_multi = np.array(kepoi_multi)



	##### GENERATE LISTS 

	P_TTVs = []
	P_plans = []
	deltaBICs = []


	single_idxs = []
	multi_idxs = []

	entrynum = 0
	for nkepoi, kepoi in enumerate(kepois):

		try:
			kepoi_period = kepoi_periods[nkepoi]
			if kepoi_period <= 10:
				### we're not interested in these!
				continue 

			print('KOI-'+str(kepoi))
			KOI_idxs = np.where(KOIs == kepoi)[0]
			if len(KOI_idxs) == 0:
				#### it's not in the catalog! Continue!
				continue

			KOI_epochs, KOI_OCs, KOI_OCerrs = epochs[KOI_idxs], OCmin[KOI_idxs], OCmin_err[KOI_idxs]
			KOI_rms = np.sqrt(np.nanmean(KOI_OCs**2))

			non_outlier_idxs = np.where(np.abs(KOI_OCs) <= 5*KOI_rms)[0]
			#non_outlier_idxs = np.where((np.abs(KOI_OCs) - 10*KOI_OCerrs) > KOI_rms)[0]
			KOI_epochs, KOI_OCs, KOI_OCerrs = KOI_epochs[non_outlier_idxs], KOI_OCs[non_outlier_idxs], KOI_OCerrs[non_outlier_idxs]
			KOI_rms = np.sqrt(np.nanmean(KOI_OCs**2))



			### run a Lomb-Scargle periodogram on this -- let the range be 2-500 epochs, 5000 logarithmically spaced bins.

			LSperiods = np.logspace(np.log10(2), np.log10(500), 5000)
			LSfreqs = 1/LSperiods
			LSpowers = LombScargle(KOI_epochs, KOI_OCs, KOI_OCerrs).power(LSfreqs)
			peak_power_idx = np.nanargmax(LSpowers)
			peak_power_period = LSperiods[peak_power_idx]
			peak_power_freq = 1/peak_power_period


			if show_plots == 'y':
				plt.plot(LSperiods, LSpowers, color='DodgerBlue', alpha=0.7, linewidth=2)
				plt.xlabel('Period [epochs]')
				plt.ylabel('Power')
				plt.xscale('log')
				plt.title('KOI-'+str(kepoi))
				plt.show()


			#### NOW FIT A SINUSOID! ### FIX THE FREQUENCY -- have to define the function anew each step to do the curve fit

			def sinecurve(tvals, amplitude, phase):
				angfreq = 2 * np.pi * peak_power_freq
				sinewave = amplitude * np.sin(angfreq * tvals + phase)
				return sinewave

			#### NOW FIT THAT SUCKER
			popt, pcov = curve_fit(sinecurve, KOI_epochs, KOI_OCs, sigma=KOI_OCerrs, bounds=([0, -2*np.pi], [20*KOI_rms, 2*np.pi]))

			

			#### calculate BIC and deltaBIC
			BIC_flat = chisquare(KOI_OCs, np.linspace(0,0,len(KOI_OCs)),KOI_OCerrs) #k = 2
			BIC_curve = 2*np.log(len(KOI_OCs)) + chisquare(KOI_OCs, sinecurve(KOI_epochs, *popt), KOI_OCerrs)
			### we want BIC_curve to be SMALLER THAN BIC_flat, despite the penalty, for the SINE MODEL TO HOLD WATER.
			#### SO IF THAT'S THE CASE, AND WE DO BIC_curve - BIC_flat, then delta-BIC will be negative, which is what we want.
			deltaBIC = BIC_curve - BIC_flat 
			deltaBICs.append(deltaBIC)


			if show_plots == 'y':
				KOI_epochs_interp = np.linspace(np.nanmin(KOI_epochs), np.nanmax(KOI_epochs), 1000)
				KOI_TTV_interp = sinecurve(KOI_epochs_interp, *popt)

				plt.scatter(KOI_epochs, KOI_OCs, facecolor='LightCoral', edgecolor='k', alpha=0.7, zorder=2)
				plt.errorbar(KOI_epochs, KOI_OCs, yerr=KOI_OCerrs, ecolor='k', fmt='none', zorder=1)
				plt.plot(KOI_epochs_interp, KOI_TTV_interp, color='k', linestyle='--', linewidth=2)
				plt.plot(KOI_epochs, np.linspace(0,0,len(KOI_epochs)), color='k', linestyle=':', alpha=0.5, zorder=0)
				plt.xlabel("epoch")
				plt.ylabel('O - C [min]')
				plt.title('KOI-'+str(kepoi)+r', $\Delta \mathrm{BIC} = $'+str(round(deltaBIC, 2)))
				plt.show()

			if deltaBIC <= -2:
				P_TTVs.append(peak_power_period)
				P_plans.append(kepoi_period)

				if kepoi_multi[nkepoi] == True:
					#multi_idxs.append(nkepoi)
					multi_idxs.append(entrynum)
				elif kepoi_multi[nkepoi] == False:
					#single_idxs.append(nkepoi)
					single_idxs.append(entrynum)

				entrynum += 1



		except:
			traceback.print_exc()
			time.sleep(5)

	multi_idxs = np.array(multi_idxs)
	single_idxs = np.array(single_idxs)
	P_TTVs = np.array(P_TTVs)
	P_plans = np.array(P_plans)

	kdestack = np.vstack((P_plans, P_TTVs))

	gkde = gaussian_kde(kdestack)
	gkde_points_p, gkde_points_t = [], []
	for p in np.logspace(np.log10(np.nanmin(P_plans)), np.log10(np.nanmax(P_plans)), 100):
		for t in np.logspace(np.log10(np.nanmin(P_TTVs)), np.log10(np.nanmax(P_TTVs)), 100):
			gkde_points_p.append(p)
			gkde_points_t.append(t)

	gkde_points_p = np.array(gkde_points_p)
	gkde_points_t = np.array(gkde_points_t)

	gkde_points = np.vstack((gkde_points_p, gkde_points_t))

	gkde_values = gkde.evaluate(gkde_points)
	gkde_norm_values = (gkde_values - np.nanmin(gkde_values)) / (np.nanmax(gkde_values) - np.nanmin(gkde_values))


	colors = cm.coolwarm(gkde_norm_values)

	plt.scatter(gkde_points_p, gkde_points_t, facecolor=colors, s=100)
	#plt.show()
	#plt.scatter(P_plans, P_TTVs, facecolor='DodgerBlue', alpha=0.5, edgecolor='k', s=20)
	plt.scatter(P_plans, P_TTVs, facecolor='k', alpha=0.5, edgecolor='k', s=5)
	plt.xlabel('planet period')
	plt.ylabel('TTV period [epochs]')
	plt.xscale('log')
	plt.yscale('log')
	plt.show()



	#### SOMETHING IS WEIRD ABOUT THE GKDE -- try a heatmap (hist2d)
	xbins = np.logspace(np.log10(10), np.log10(1500), 20) #### consistent with the simulation
	ybins = np.logspace(np.log10(2), np.log10(100), 20) #### consistent with the simulation
	TTV_Pplan_hist2d = np.histogram2d(P_plans, P_TTVs, bins=[xbins, ybins])
	plt.imshow(TTV_Pplan_hist2d[0].T, origin='lower', cmap=cm.coolwarm)
	plt.xticks(ticks=np.arange(0,len(xbins),5), labels=np.around(np.log10(xbins[::5]),2))
	plt.yticks(ticks=np.arange(0,len(ybins),5), labels=np.around(np.log10(ybins[::5]), 2))
	plt.xlabel(r'$\log_{10} \, P_{\mathrm{P}}$ [days]')
	plt.ylabel(r'$\log_{10} \, P_{\mathrm{TTV}}$ [epochs]')
	plt.tight_layout()
	plt.show()

	np.save('/data/tethys/Documents/Projects/NMoon_TTVs/mazeh_PTTV10-1500_Pplan2-100_20x20_heatmap.npy', TTV_Pplan_hist2d)


	#### COMPARE TO NATIVE MATPLOTLIB HISTOGRAM
	#### THIS IS MUCH BETTER -- you get the tick labels for free... could even do a scatter over top
	plt.figure(figsize=(6,6))
	heatmap = plt.hist2d(P_plans, P_TTVs, bins=[xbins, ybins], cmap='coolwarm')[0]
	plt.scatter(P_plans, P_TTVs, facecolor='w', edgecolor='k', s=5, alpha=0.3)
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel(r'$P_{\mathrm{P}}$ [days]')
	plt.ylabel(r'$P_{\mathrm{TTV}}$ [epochs]')
	#plt.title('Matplotlib 2D histogram')
	plt.show()


	##### LOOK AT THE HEATMAP FOR SINGLES AND MULTIS
	#### COMPARE TO NATIVE MATPLOTLIB HISTOGRAM
	#### THIS IS MUCH BETTER -- you get the tick labels for free... could even do a scatter over top
	fig, (ax1, ax2) = plt.subplots(2, figsize=(6,12))
	heatmap_single = ax1.hist2d(P_plans[single_idxs], P_TTVs[single_idxs], bins=[xbins, ybins], cmap='coolwarm', density=False)[0]
	ax1.scatter(P_plans[single_idxs], P_TTVs[single_idxs], facecolor='w', edgecolor='k', s=5, alpha=0.3)
	ax1.set_xscale('log')
	ax1.set_yscale('log')
	ax1.set_ylabel(r'$P_{\mathrm{TTV}}$ [epochs] (single)')

	heatmap_multi = ax2.hist2d(P_plans[multi_idxs], P_TTVs[multi_idxs], bins=[xbins, ybins], cmap='coolwarm', density=False)[0]
	ax2.scatter(P_plans[multi_idxs], P_TTVs[multi_idxs], facecolor='w', edgecolor='k', s=5, alpha=0.3)
	ax2.set_xscale('log')
	ax2.set_yscale('log')
	ax2.set_ylabel(r'$P_{\mathrm{TTV}}$ [epochs] (multi)')

	ax2.set_xlabel(r'$P_{\mathrm{P}}$ [days]')


	#plt.title('Matplotlib 2D histogram')
	plt.show()	





	heatmap_column_sums = []
	for i in np.arange(0,19,1):
		heatmap_colsum = np.nansum(heatmap[i])
		heatmap_column_sums.append(heatmap_colsum)
	heatmap_column_sums = np.array(heatmap_column_sums)

	heatmap_div_pplans = (heatmap.T / heatmap_column_sums).T

	##### UNFORTUNATELY YOU CAN'T USE THE SAME FRAMEWORK AS BEFORE!
	







except:
	traceback.print_exc()



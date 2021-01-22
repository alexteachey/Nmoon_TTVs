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
import matplotlib.gridspec as gridspec 
from scipy.special import factorial
from moonpy import *
import socket
from mr_forecast import Rstat2M
from scipy.stats import kstest 

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



def TTVkiller(rms_amp, errors, Pplan, Tdur, unit='days', npoints=None):
	### equation 20 from this paper: https://arxiv.org/pdf/2004.04230.pdf
	"""
	Errors should be array-like, or npoints must be specified.
	Pplan and Tdur need to have the same units, as shoudl rms_amp and errors.

	"""
	LHS_num = rms_amp
	errorbar = np.nanmedian(errors)
	try:
		npoints = len(errors)
	except:
		pass
	LHS_denom = (np.sqrt(2) * errorbar) / np.sqrt(npoints) 
	LHS = LHS_num / LHS_denom

	RHS_first_term = np.sqrt(3) / (4*np.pi)
	RHS_second_term = Pplan / Tdur 
	RHS = RHS_first_term * RHS_second_term 

	if LHS < RHS:
		moon_possible = True
	else:
		moon_possible = False



def fmin(mplan, mstar, A_TTV, Pplan, unit='minutes'):
	#### units of A_TTV and Pplan have to be the same!

	q = mplan / mstar 
	first_term = 9 / (q**(1/3))
	second_term = A_TTV / Pplan 
	minimum_fRHill = first_term * second_term
	return minimum_fRHill 
	



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
	kepler_radius_rearth = np.array(cumkois['koi_prad'])
	kepler_radius_rearth_uperr = np.array(cumkois['koi_prad_err1'])
	kepler_radius_rearth_lowerr = np.array(cumkois['koi_prad_err2'])
	kepler_radius_rearth_err = np.nanmean((kepler_radius_rearth_uperr, np.abs(kepler_radius_rearth_lowerr)), axis=0)


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
				HLplanet_proper_format = HLplanet[:-1]+' '+HLplanet[-1]
				if HLplanet_proper_format.startswith('Kepler-25') or HLplanet_proper_format.startswith('Kepler-89') or HLplanet_proper_format.startswith('Kepler-444'):
					HLplanet_proper_format = HLplanet_proper_format[:-2]+' A '+HLplanet_proper_format[-1]
				HL_kepoi_idx = np.where(kepler_names == HLplanet_proper_format)[0]
				HL_kepoi = kepois[HL_kepoi_idx]
			else:
				HL_kepoi = HLplanet #### of the form K0001.01, etc

			if type(HL_kepoi) == np.ndarray:
				HL_kepoi = HL_kepoi[0]
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
	kepoi_multi_Pip1_over_Pis = [] #### P_{i+1} / P_i #### covers all but the outermost planet
	kepoi_multi_Pi_over_Pim1s = [] #### P_i / P_{i - 1} #### cover all but the innermost planet

	for nkep,kep in enumerate(kepois): ### kepois are already strings
		kepoi_number = kep[:kep.find('.')] #### leaves off the final .01, .02, .03, etc.
		#### find how many entries match this in system_nums.
		all_system_planet_idxs = np.where(kepoi_number == system_nums)[0]
		all_system_planet_periods = kepoi_periods[all_system_planet_idxs]
		all_system_planet_periods_sorted = np.sort(all_system_planet_periods)
		this_planet_idx = np.where(kep == kepois)[0]
		this_planet_period = kepoi_periods[this_planet_idx][0]
		this_planet_order_number_idx = np.where(all_system_planet_periods_sorted == this_planet_period)[0]
		try:
			next_highest_period = all_system_planet_periods_sorted[this_planet_order_number_idx+1]
		except:
			next_highest_period = np.nan 

		try:
			if this_planet_order_number_idx != 0:
				next_lowest_period = all_system_planet_periods_sorted[this_planet_order_number_idx-1]
			else:
				next_lowest_period = np.nan
		except:
			next_lowest_period = np.nan 


		this_planet_Pip1_over_Pi = next_highest_period / this_planet_period
		this_planet_Pi_over_Pim1 = this_planet_period / next_lowest_period

		assert (this_planet_Pip1_over_Pi >= 1) or (np.isfinite(this_planet_Pip1_over_Pi) == False)
		assert (this_planet_Pi_over_Pim1 >= 1) or (np.isfinite(this_planet_Pi_over_Pim1) == False)

		kepoi_multi_Pip1_over_Pis.append(this_planet_Pip1_over_Pi)
		kepoi_multi_Pi_over_Pim1s.append(this_planet_Pi_over_Pim1)


		#

		nplanets_in_system = len(all_system_planet_idxs)
		if nplanets_in_system > 1:
			kepoi_multi.append(True)

		elif nplanets_in_system == 1:
			kepoi_multi.append(False)
		else:
			raise Exception('something weird happening with the single / multi counter.')
	kepoi_multi = np.array(kepoi_multi)
	kepoi_multi_Pip1_over_Pis = np.array(kepoi_multi_Pip1_over_Pis)
	kepoi_multi_Pi_over_Pim1s = np.array(kepoi_multi_Pi_over_Pim1s)








	##### GENERATE LISTS 

	radii = [] #### EARTH RADII
	radii_errors = []
	P_TTVs = []
	P_plans = []
	deltaBICs = []
	Pip1_over_Pis = []
	Pi_over_Pim1s = []

	TTV_amplitudes = []
	single_idxs = []
	multi_idxs = []
	in_HLcatalog_idxs = []
	notin_HLcatalog_idxs = []


	entrynum = 0
	for nkepoi, kepoi in enumerate(kepois):

		try:
			kepoi_period = kepoi_periods[nkepoi]
			#if kepoi_period <= 10:
			#	### we're not interested in these!
			#	continue 

			print('KOI-'+str(kepoi))
			KOI_idxs = np.where(KOIs == kepoi)[0]
			if len(KOI_idxs) == 0:
				#### it's not in the catalog! Continue!
				continue

			KOI_epochs, KOI_OCs, KOI_OCerrs = epochs[KOI_idxs], OCmin[KOI_idxs], OCmin_err[KOI_idxs]
			KOI_rms = np.sqrt(np.nanmean(KOI_OCs**2))
			orig_KOI_rms = KOI_rms

			non_outlier_idxs = np.where(np.abs(KOI_OCs) <= np.sqrt(2)*KOI_rms)[0]
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


			"""
			if show_plots == 'y':
				plt.plot(LSperiods, LSpowers, color='DodgerBlue', alpha=0.7, linewidth=2)
				plt.xlabel('Period [epochs]')
				plt.ylabel('Power')
				plt.xscale('log')
				plt.title('KOI-'+str(kepoi))
				plt.show()
			"""

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


			if (show_plots == 'y') and ('KOI-'+kepoi in HL_KOIs):
				KOI_epochs_interp = np.linspace(np.nanmin(KOI_epochs), np.nanmax(KOI_epochs), 1000)
				KOI_TTV_interp = sinecurve(KOI_epochs_interp, *popt)

				plt.scatter(KOI_epochs, KOI_OCs, facecolor='LightCoral', edgecolor='k', alpha=0.7, zorder=2)
				plt.errorbar(KOI_epochs, KOI_OCs, yerr=KOI_OCerrs, ecolor='k', fmt='none', zorder=1)
				plt.plot(KOI_epochs_interp, KOI_TTV_interp, color='k', linestyle='--', linewidth=2)
				plt.plot(KOI_epochs, np.linspace(0,0,len(KOI_epochs)), color='k', linestyle=':', alpha=0.5, zorder=0)
				plt.xlabel("epoch")
				plt.ylabel('O - C [min]')
				plt.title('KOI-'+str(kepoi)+r', rms = '+str(round(orig_KOI_rms,2))+', $\Delta \mathrm{BIC} = $'+str(round(deltaBIC, 2)))
				plt.show()

			if deltaBIC <= -2:
				radii.append(kepler_radius_rearth[nkepoi])
				radii_errors.append(kepler_radius_rearth_err[nkepoi])
				P_TTVs.append(peak_power_period)
				P_plans.append(kepoi_period)
				amplitude = np.nanmax(np.abs(BIC_curve))
				TTV_amplitudes.append(amplitude)
				Pip1_over_Pis.append(kepoi_multi_Pip1_over_Pis)
				Pi_over_Pim1s.append(kepoi_multi_Pi_over_Pim1s)

				if kepoi_multi[nkepoi] == True:
					#multi_idxs.append(nkepoi)
					multi_idxs.append(entrynum)
				elif kepoi_multi[nkepoi] == False:
					#single_idxs.append(nkepoi)
					single_idxs.append(entrynum)


				if 'KOI-'+kepoi in HL_KOIs:
					in_HLcatalog_idxs.append(entrynum)
				else:
					notin_HLcatalog_idxs.append(entrynum)

				entrynum += 1



		except:
			traceback.print_exc()
			time.sleep(5)

	multi_idxs = np.array(multi_idxs)
	single_idxs = np.array(single_idxs)
	in_HLcatalog_idxs = np.array(in_HLcatalog_idxs)
	notin_HLcatalog_idxs = np.array(notin_HLcatalog_idxs)
	notin_HLcatalog_single_idxs = np.intersect1d(single_idxs, notin_HLcatalog_idxs)
	notin_HLcatalog_multi_idxs = np.intersect1d(multi_idxs, notin_HLcatalog_idxs)
	TTV_amplitudes = np.array(TTV_amplitudes)

	print('# singles , # multis = ', len(single_idxs), len(multi_idxs))
	print('# in HL , # not in HL = ', len(in_HLcatalog_idxs), len(notin_HLcatalog_idxs))

	P_TTVs = np.array(P_TTVs)
	P_plans = np.array(P_plans)
	radii = np.array(radii)
	radii_errors = np.array(radii_errors)



	### generate masses
	"""
	Rstat2M = np.vectorize(Rstat2M)
	print('FORECASTing masses from radii: ')
	forecast_masses, forecast_masses_uperr, forecast_masses_lowerr = Rstat2M(mean=radii, std=radii_errors, unit='Earth')

	highest_mass = np.nanmax(forecast_masses)
	normalized_masses = forecast_masses / highest_mass

	amplitudes_div_masses = TTV_amplitudes / normalized_masses
	"""

	plt.scatter(P_plans[notin_HLcatalog_single_idxs], TTV_amplitudes[notin_HLcatalog_single_idxs], facecolor='green', edgecolor='k', s=20, alpha=0.5, label='single non-HL2017')
	plt.scatter(P_plans[notin_HLcatalog_multi_idxs], TTV_amplitudes[notin_HLcatalog_multi_idxs], facecolor='DodgerBlue', edgecolor='k', s=20, alpha=0.5, label='multi non-HL2017')
	plt.scatter(P_plans[in_HLcatalog_idxs], TTV_amplitudes[in_HLcatalog_idxs], facecolor='LightCoral', edgecolor='k', s=20, alpha=0.5, label='multi HL2017')
	plt.xlabel(r'$P_{\mathrm{P}}$ [days]')
	plt.ylabel('TTV amplitude [s]')
	plt.yscale('log')
	plt.legend()
	plt.show()


	plt.scatter(P_TTVs[notin_HLcatalog_single_idxs], TTV_amplitudes[notin_HLcatalog_single_idxs], facecolor='green', edgecolor='k', s=20, alpha=0.5, label='single non-HL2017')
	plt.scatter(P_TTVs[notin_HLcatalog_multi_idxs], TTV_amplitudes[notin_HLcatalog_multi_idxs], facecolor='DodgerBlue', edgecolor='k', s=20, alpha=0.5, label='multi non-HL2017')
	plt.scatter(P_TTVs[in_HLcatalog_idxs], TTV_amplitudes[in_HLcatalog_idxs], facecolor='LightCoral', edgecolor='k', s=20, alpha=0.5, label='multi HL2017')
	plt.xlabel(r'$P_{\mathrm{TTV}}$ [epochs]')
	plt.ylabel('TTV amplitude [s]')
	plt.yscale('log')
	plt.xscale('log')
	plt.legend()
	plt.show()


	plt.scatter(P_plans[notin_HLcatalog_single_idxs], P_TTVs[notin_HLcatalog_single_idxs], facecolor='green', edgecolor='k', s=20, alpha=0.5, label='single non-HL2017')
	plt.scatter(P_plans[notin_HLcatalog_multi_idxs], P_TTVs[notin_HLcatalog_multi_idxs], facecolor='DodgerBlue', edgecolor='k', s=20, alpha=0.5, label='multi non-HL2017')
	plt.scatter(P_plans[in_HLcatalog_idxs], P_TTVs[in_HLcatalog_idxs], facecolor='LightCoral', edgecolor='k', s=20, alpha=0.5, label='multi HL2017')
	plt.xlabel(r'$P_{\mathrm{P}}$ [days]')
	plt.ylabel(r'$P_{\mathrm{TTV}}$ [epochs]')
	plt.yscale('log')
	plt.xscale('log')
	plt.legend()
	plt.show()


	"""
	#### replace all forecast_masses == 0.0 with np.nan!
	forecast_masses[np.where(forecast_masses == 0.0)[0]] = np.nan

	#### look at the mass distributions 
	plt.scatter(P_plans[notin_HLcatalog_single_idxs], forecast_masses[notin_HLcatalog_single_idxs], facecolor='green', edgecolor='k', s=20, alpha=0.5, label='single non-HL2017')
	plt.scatter(P_plans[notin_HLcatalog_multi_idxs], forecast_masses[notin_HLcatalog_multi_idxs], facecolor='DodgerBlue', edgecolor='k', s=20, alpha=0.5, label='multi non-HL2017')
	plt.scatter(P_plans[in_HLcatalog_idxs], forecast_masses[in_HLcatalog_idxs], facecolor='LightCoral', edgecolor='k', s=20, alpha=0.5, label='multi HL2017')
	


	plt.xlabel(r'$P_{\mathrm{P}}$ [days]')
	plt.ylabel(r'FORECASTER $M_{\oplus}$')
	plt.yscale('log')
	plt.xscale('log')
	plt.legend()
	plt.show()
	"""






	#raise Exception('this is all you want to do right now.')



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
	xcenters, ycenters = [], []
	for nxb,xb in enumerate(xbins):
		try:
			xcenters.append(np.nanmean((xbins[nxb+1], xbins[nxb])))
		except:
			pass

	for nyb, yb in enumerate(ybins):
		try:
			ycenters.append(np.nanmean((ybins[nyb+1], ybins[nyb])))
		except:
			pass
	xcenters, ycenters = np.array(xcenters), np.array(ycenters)



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

	np.save('/data/tethys/Documents/Projects/NMoon_TTVs/heatmap_single.npy', heatmap_single)	

	heatmap_multi = ax2.hist2d(P_plans[multi_idxs], P_TTVs[multi_idxs], bins=[xbins, ybins], cmap='coolwarm', density=False)[0]
	ax2.scatter(P_plans[multi_idxs], P_TTVs[multi_idxs], facecolor='w', edgecolor='k', s=5, alpha=0.3)
	ax2.set_xscale('log')
	ax2.set_yscale('log')
	ax2.set_ylabel(r'$P_{\mathrm{TTV}}$ [epochs] (multi)')

	ax2.set_xlabel(r'$P_{\mathrm{P}}$ [days]')


	#plt.title('Matplotlib 2D histogram')
	plt.show()	




	##### LOOK AT THE DISTRIBUTION FOR HL2017 SOURCES and Non-HL2017 sources
	#### COMPARE TO NATIVE MATPLOTLIB HISTOGRAM
	#### THIS IS MUCH BETTER -- you get the tick labels for free... could even do a scatter over top
	fig, (ax1, ax2) = plt.subplots(2, figsize=(6,12))
	heatmap_single = ax1.hist2d(P_plans[in_HLcatalog_idxs], P_TTVs[in_HLcatalog_idxs], bins=[xbins, ybins], cmap='coolwarm', density=False)[0]
	ax1.scatter(P_plans[in_HLcatalog_idxs], P_TTVs[in_HLcatalog_idxs], facecolor='w', edgecolor='k', s=5, alpha=0.3)
	ax1.set_xscale('log')
	ax1.set_yscale('log')
	ax1.set_ylabel(r'$P_{\mathrm{TTV}}$ [epochs] (HL2017)')

	heatmap_multi = ax2.hist2d(P_plans[notin_HLcatalog_idxs], P_TTVs[notin_HLcatalog_idxs], bins=[xbins, ybins], cmap='coolwarm', density=False)[0]
	ax2.scatter(P_plans[notin_HLcatalog_idxs], P_TTVs[notin_HLcatalog_idxs], facecolor='w', edgecolor='k', s=5, alpha=0.3)
	ax2.set_xscale('log')
	ax2.set_yscale('log')
	ax2.set_ylabel(r'$P_{\mathrm{TTV}}$ [epochs] (non-HL2017)')

	ax2.set_xlabel(r'$P_{\mathrm{P}}$ [days]')


	#plt.title('Matplotlib 2D histogram')
	plt.show()	



	##### LOOK AT AMPLITUDE VS PTTV FOR HL2017 SOURCES and Non-HL2017 sources
	#### COMPARE TO NATIVE MATPLOTLIB HISTOGRAM
	#### THIS IS MUCH BETTER -- you get the tick labels for free... could even do a scatter over top
	fig, (ax1, ax2) = plt.subplots(2, figsize=(6,12))
	heatmap_single = ax1.hist2d(P_TTVs[in_HLcatalog_idxs], TTV_amplitudes[in_HLcatalog_idxs], bins=[ybins, np.arange(0,100,5)], cmap='coolwarm', density=False)[0]
	ax1.scatter(P_TTVs[in_HLcatalog_idxs], TTV_amplitudes[in_HLcatalog_idxs], facecolor='w', edgecolor='k', s=5, alpha=0.3)
	ax1.set_xscale('log')
	#ax1.set_yscale('log')
	ax1.set_ylabel(r'amplitude [minutes] (HL2017)')

	heatmap_multi = ax2.hist2d(P_TTVs[notin_HLcatalog_idxs], TTV_amplitudes[notin_HLcatalog_idxs], bins=[ybins, np.arange(0,100,5)], cmap='coolwarm', density=False)[0]
	ax2.scatter(P_TTVs[notin_HLcatalog_idxs], TTV_amplitudes[notin_HLcatalog_idxs], facecolor='w', edgecolor='k', s=5, alpha=0.3)
	ax2.set_xscale('log')
	#ax2.set_yscale('log')
	ax2.set_ylabel(r'amplitude [minutes] (non-HL2017)')

	ax2.set_xlabel(r'$P_{\mathrm{TTV}}$ [days]')


	#plt.title('Matplotlib 2D histogram')
	plt.show()	




	##### LOOK AT AMPLITUDE VS PTTV FOR SINGLES AND MULTIs
	#### COMPARE TO NATIVE MATPLOTLIB HISTOGRAM
	#### THIS IS MUCH BETTER -- you get the tick labels for free... could even do a scatter over top
	fig, (ax1, ax2) = plt.subplots(2, figsize=(6,12))
	heatmap_single = ax1.hist2d(P_TTVs[single_idxs], TTV_amplitudes[single_idxs], bins=[ybins, np.arange(0,100,5)], cmap='coolwarm', density=False)[0]
	ax1.scatter(P_TTVs[single_idxs], TTV_amplitudes[single_idxs], facecolor='w', edgecolor='k', s=5, alpha=0.3)
	ax1.set_xscale('log')
	#ax1.set_yscale('log')
	ax1.set_ylabel(r'amplitude [minutes] (single)')

	heatmap_multi = ax2.hist2d(P_TTVs[multi_idxs], TTV_amplitudes[multi_idxs], bins=[ybins, np.arange(0,100,5)], cmap='coolwarm', density=False)[0]
	ax2.scatter(P_TTVs[multi_idxs], TTV_amplitudes[multi_idxs], facecolor='w', edgecolor='k', s=5, alpha=0.3)
	ax2.set_xscale('log')
	#ax2.set_yscale('log')
	ax2.set_ylabel(r'amplitude [minutes] (multi)')

	ax2.set_xlabel(r'$P_{\mathrm{P}}$ [days]')


	#plt.title('Matplotlib 2D histogram')
	plt.show()	





	#### make straight-up histograms for HL2017 and non-HL20217
	fig, (ax1, ax2) = plt.subplots(2, sharex=True)
	ax1.hist(P_TTVs[in_HLcatalog_idxs], bins=ybins, facecolor='DodgerBlue', edgecolor='k', alpha=0.7)
	ax1.set_ylabel('HL2017')
	ax1.set_xscale('log')
	ax2.hist(P_TTVs[notin_HLcatalog_idxs], bins=ybins, facecolor='LightCoral', edgecolor='k', alpha=0.7)
	ax2.set_ylabel('non - HL2017')
	ax2.set_xlabel(r'$P_{\mathrm{TTV}}$ [epochs]')
	ax2.set_xscale('log')
	plt.show()



	#### make straight-up histograms for HL2017 and non-HL20217 -- AMPLITUDES
	fig, (ax1, ax2) = plt.subplots(2, sharex=True)
	ax1.hist(TTV_amplitudes[in_HLcatalog_idxs], bins=ybins, facecolor='DodgerBlue', edgecolor='k', alpha=0.7)
	ax1.set_ylabel('HL2017')
	ax1.set_xscale('log')
	ax2.hist(TTV_amplitudes[notin_HLcatalog_idxs], bins=ybins, facecolor='LightCoral', edgecolor='k', alpha=0.7)
	ax2.set_ylabel('non - HL2017')
	ax2.set_xlabel(r'TTV amplitude [minutes]')
	ax2.set_xscale('log')
	plt.show()



	#### AMPLTIUDES -- singles vs multies
	fig, (ax1, ax2) = plt.subplots(2, sharex=True)
	ax1.hist(TTV_amplitudes[single_idxs], bins=ybins, facecolor='DodgerBlue', edgecolor='k', alpha=0.7)
	ax1.set_ylabel('single')
	ax1.set_xscale('log')
	ax2.hist(TTV_amplitudes[multi_idxs], bins=ybins, facecolor='LightCoral', edgecolor='k', alpha=0.7)
	ax2.set_ylabel('multi')
	ax2.set_xlabel(r'TTV amplitude [minutes]')
	ax2.set_xscale('log')
	plt.show()


	deltaBICs = np.array(deltaBICs)

	#### do the same for Delta-BICs (HL vs nonHL)
	fig, (ax1, ax2) = plt.subplots(2, sharex=True)
	ax1.hist(deltaBICs[in_HLcatalog_idxs], bins=np.arange(-100,0,5), facecolor='DodgerBlue', edgecolor='k', alpha=0.7)
	ax1.set_ylabel('HL2017')
	#ax1.set_xscale('log')
	ax2.hist(deltaBICs[notin_HLcatalog_idxs], bins=np.arange(-100,0,5), facecolor='LightCoral', edgecolor='k', alpha=0.7)
	ax2.set_ylabel('non - HL2017')
	ax2.set_xlabel(r'$\Delta$BIC')
	#ax2.set_xscale('log')
	plt.show()



	#### do the same for Delta-BICs (single vs multi)
	fig, (ax1, ax2) = plt.subplots(2, sharex=True)
	ax1.hist(deltaBICs[single_idxs], bins=np.arange(-100,0,5), facecolor='DodgerBlue', edgecolor='k', alpha=0.7)
	ax1.set_ylabel('single')
	#ax1.set_xscale('log')
	ax2.hist(deltaBICs[multi_idxs], bins=np.arange(-100,0,5), facecolor='LightCoral', edgecolor='k', alpha=0.7)
	ax2.set_ylabel('multi')
	ax2.set_xlabel(r'$\Delta$BIC')
	#ax2.set_xscale('log')
	plt.show()



	##### PLOT amplitude and PTTV as a function of the MULTI periods

	fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
	ax1.scatter(kepoi_multi_Pip1_over_Pis[multi_idxs], TTV_amplitudes[multi_idxs], facecolor='DodgerBlue', edgecolor='k', s=20, alpha=0.7)
	ax1.set_ylabel('TTV amplitude [minutes]')
	ax1.set_xlabel(r'$P_{i+1} / P_i$ ')
	ax1.set_xscale('log')
	ax1.set_yscale('log')
	#ax1.set_ylim(0,100)
	ax2.scatter(kepoi_multi_Pi_over_Pim1s[multi_idxs], TTV_amplitudes[multi_idxs], facecolor='LightCoral', edgecolor='k', s=20, alpha=0.7)
	ax2.set_ylabel('TTV amplitude [minutes]')
	ax2.set_xlabel(r'$P_i / P_{i-1}$ ')
	ax2.set_xscale('log')
	ax2.set_yscale('log')
	#ax2.set_ylim(0,1000)
	plt.show()



	##### CREATE ONE GIANT PANEL -- (4x4) with 1) sims, 2) non-HL singles, 3) non-HL multis, 4) HL, and 
	##### Pplan, PTTV, Amplitudes, DeltaBICs

	#np.save(projectdir+'/sim_deltaBIC_list.npy', deltaBIC_list[good_BIC_stable_idxs])
	#np.save(projectdir+'/sim_PTTVs.npy', P_TTVs)
	#np.save(projectdir+'/sim_Pplans.npy', P_plans)

	try:
		sim_deltaBIC_list = np.load('/data/tethys/Documents/Projects/NMoon_TTVs/sim_deltaBIC_list.npy')
		sim_PTTVs = np.load('/data/tethys/Documents/Projects/NMoon_TTVs/sim_PTTVs.npy')
		sim_Pplans = np.load('/data/tethys/Documents/Projects/NMoon_TTVs/sim_Pplans.npy')
		sim_TTV_amplitudes = np.load('/data/tethys/Dcuments/Projects/NMoon_TTVs/sim_TTV_amplitudes.npy')

	except:
		sim_deltaBIC_list = np.load(projectdir+'/sim_deltaBIC_list.npy')
		sim_PTTVs = np.load(projectdir+'/sim_PTTVs.npy')
		sim_Pplans = np.load(projectdir+'/sim_Pplans.npy')
		sim_TTV_amplitudes = np.load(projectdir+'/sim_TTV_amplitudes.npy')


		#try:
		#	sim_TTV_ampltiudes = np.load(projectdir+'/sim_TTV_amplitudes.npy')
		#except:
		#	sim_TTV_amplitudes = np.random.randint(low=0, high=1e3, size=len(sim_deltaBIC_list))+np.random.random(size=len(sim_deltaBIC_list))


	sim_TTV_amplitudes_minutes = sim_TTV_amplitudes / 60

	single_notHL_idxs = np.intersect1d(notin_HLcatalog_idxs, single_idxs)
	multi_notHL_idxs = np.intersect1d(notin_HLcatalog_idxs, multi_idxs)
	multi_HL_idxs = np.intersect1d(in_HLcatalog_idxs, multi_idxs) #### should be the same as in_HLcatalog_idxs

	deltaBIC_lists = [sim_deltaBIC_list, deltaBICs[single_notHL_idxs], deltaBICs[multi_notHL_idxs], deltaBICs[multi_HL_idxs]]
	deltaBIC_bins = np.linspace(-100,-2,20)
	
	PTTV_lists = [sim_PTTVs, P_TTVs[single_notHL_idxs], P_TTVs[multi_notHL_idxs], P_TTVs[multi_HL_idxs]]
	PTTV_bins = np.logspace(np.log10(2), np.log10(100), 20)
	
	TTVamp_lists = [sim_TTV_amplitudes_minutes, TTV_amplitudes[single_notHL_idxs], TTV_amplitudes[multi_notHL_idxs], TTV_amplitudes[multi_HL_idxs]]
	TTVamp_bins = np.logspace(0,6,20)

	#PTTV_over_Pplan_lists = [sim_PTTVs / sim_Pplans, P_TTVs[single_notHL_idxs]/P_plans[single_notHL_idxs], P_TTVs[multi_notHL_idxs]/P_plans[multi_notHL_idxs], P_TTVs[multi_HL_idxs] / P_plans[multi_HL_idxs]]
	#PTTV_over_Pplan_bins = np.logspace(0,3,20)

	Pplan_lists = [sim_Pplans, P_plans[single_notHL_idxs], P_plans[multi_notHL_idxs], P_plans[multi_HL_idxs]]
	Pplan_bins = np.logspace(np.log10(1), np.log10(1500), 20)
	#amp_over_mass_lists = [sim_AoverM, amplitudes_div_masses[single_notHL_idxs], amplitudes_div_masses[multi_notHL_idxs], amplitudes_div_masses[multi_HL_idxs]]


	row_labels = ['moon sims', 'singles', 'multis', 'HL2017']
	col_labels = [r'$P_{\mathrm{P}}$ [days]', r'$P_{\mathrm{TTV}}$ [epochs]', 'TTV amplitude [min]', r'$\Delta$BIC']
	column_list_of_lists = [Pplan_lists, PTTV_lists, TTVamp_lists, deltaBIC_lists]
	bin_lists = [Pplan_bins, PTTV_bins, TTVamp_bins, deltaBIC_bins]
	axis_scales = ['log', 'log', 'log', 'linear']

	colors = cm.viridis(np.linspace(0,1,len(bin_lists)))	

	nrows = 4 #### sim, single, multi, HL
	ncols = 4 #### Pplan, PTTV, deltaBIC

	fig, ax = plt.subplots(nrows,ncols) ### might need to reverse this
	#plt.figure(figsize=(8,8))
	#gs1 = gridspec.Gridspec(nrows,ncols)
	#gs1.update(hspace=0.05)

	### coordinates are row (top=0), then column (left=0)
	### rows are sim, non-HL single, non-HL multi, HL
	### columns are Pplan, PTTV, DeltaBIC

	for row in np.arange(0,nrows,1):
		for col in np.arange(0,ncols,1):

			ax[row][col].hist(column_list_of_lists[col][row], bins=bin_lists[col], facecolor=colors[col], edgecolor='k', alpha=0.5)
			ax[row][col].set_xscale(axis_scales[col]) 
			if col == ncols-1:
				ax[row][col].yaxis.set_label_position("right")
				#ax[row][col].yaxis.tick_right()
				ax[row][col].set_ylabel(row_labels[row])

			if row == nrows-1:
				ax[row][col].set_xlabel(col_labels[col])

			if row != nrows-1:
				ax[row][col].set_xticklabels([])

			#if col != 0:
			#	ax[row][col].set_yticklabels([])

			#ax[row][col].set_yticklabels([])

	plt.tight_layout()
	plt.show()




	#### let's compute a bunch of KS values.
	#### DO IT FOR HL against ALL, and HL against MULTIS.
	#### values we will test are P_TTVs, deltaBICs, P_plans, 


	ks_PTTV_allvsHL = kstest(P_TTVs[notin_HLcatalog_idxs], P_TTVs[in_HLcatalog_idxs])
	ks_PTTV_multivsHL = kstest(P_TTVs[notin_HLcatalog_multi_idxs], P_TTVs[in_HLcatalog_idxs])
	ks_Pplan_allvsHL = kstest(P_plans[notin_HLcatalog_idxs], P_plans[in_HLcatalog_idxs])
	ks_Pplan_multivsHL = kstest(P_plans[notin_HLcatalog_multi_idxs], P_plans[in_HLcatalog_idxs])
	ks_TTVamp_allvsHL = kstest(TTV_amplitudes[notin_HLcatalog_idxs], TTV_amplitudes[in_HLcatalog_idxs])
	ks_TTVamp_multivsHL = kstest(TTV_amplitudes[notin_HLcatalog_multi_idxs], TTV_amplitudes[in_HLcatalog_idxs])
	#ks_forecast_masses_allvsHL = kstest(forecast_masses[notin_HLcatalog_idxs], forecast_masses[in_HLcatalog_idxs])
	#ks_forecast_masses_multivsHL = kstest(forecast_masses[notin_HLcatalog_multi_idxs], forecast_masses[in_HLcatalog_idxs])
	ks_radii_allvsHL = kstest(radii[notin_HLcatalog_idxs], radii[in_HLcatalog_idxs])
	ks_radii_multivsHL = kstest(radii[notin_HLcatalog_multi_idxs], radii[in_HLcatalog_idxs])	

	ks_Pip1_multivsHL = kstest(Pip1_over_Pis[notin_HLcatalog_idxs], Pip1_over_Pis[in_HLcatalog_idxs])
	ks_Pi_over_Pim1s = kstest(Pi_over_Pim1s[notin_HLcatalog_idxs], Pi_over_Pim1s[in_HLcatalog_idxs])





	sim_PTTVs = np.load('/run/media/amteachey/Auddy_Akiti/Teachey/Nmoon_TTVs/sim_PTTVs.npy')
	sim_Pplans = np.load('/run/media/amteachey/Auddy_Akiti/Teachey/Nmoon_TTVs/sim_Pplans.npy')






	#### 4 panel heatmaps

	fig, ((ax1,ax2), (ax3, ax4)) = plt.subplots(nrows=2,ncols=2)
	sim_heatmap = ax1.hist2d(sim_Pplans, sim_PTTVs, bins=[xbins, ybins], cmap='coolwarm', density=False)[0]
	ax1.scatter(sim_Pplans, sim_PTTVs, facecolor='w', edgecolor='k', alpha=0.2, s=10)
	ax1.set_xscale('log')	
	ax1.set_yscale('log')
	
	single_heatmap = ax2.hist2d(P_plans[single_idxs], P_TTVs[single_idxs], bins=[xbins, ybins], cmap='coolwarm', density=False)[0]
	ax2.scatter(P_plans[single_idxs], P_TTVs[single_idxs], facecolor='w', edgecolor='k', alpha=0.2, s=10)
	ax2.set_xscale('log')
	ax2.set_yscale('log')

	multi_nonHL_heatmap = ax3.hist2d(P_plans[notin_HLcatalog_multi_idxs], P_TTVs[notin_HLcatalog_multi_idxs], bins=[xbins, ybins], cmap='coolwarm', density=False)[0]
	ax3.scatter(P_plans[notin_HLcatalog_multi_idxs], P_TTVs[notin_HLcatalog_multi_idxs], facecolor='w', edgecolor='k', alpha=0.2, s=10)
	ax3.set_xscale('log')
	ax3.set_yscale('log')

	multi_HL_heatmap = ax4.hist2d(P_plans[in_HLcatalog_idxs], P_TTVs[in_HLcatalog_idxs], bins=[xbins, ybins], cmap='coolwarm', density=False)[0]
	ax4.scatter(P_plans[in_HLcatalog_idxs], P_TTVs[in_HLcatalog_idxs], facecolor='w', edgecolor='k', alpha=0.2, s=10)
	ax4.set_xscale('log')
	ax4.set_yscale('log')
	plt.show()



	#### 
	sim_heatmap_frac_of_Pplan = np.zeros(shape=sim_heatmap.shape)
	single_heatmap_frac_of_Pplan = np.zeros(shape=single_heatmap.shape)
	multi_nonHL_heatmap_frac_of_Pplan = np.zeros(shape=multi_nonHL_heatmap.shape)
	multi_HL_heatmap_frac_of_Pplan = np.zeros(shape=multi_HL_heatmap.shape)

	#### these will all be the same shape, so
	nrows, ncols = sim_heatmap.shape

	for col in np.arange(0,ncols,1):
		#### divide each cell in the column by the sum of that column
		#sim_heatmap_frac_of_Pplan[col] = sim_heatmap[col] / np.nansum(sim_heatmap[col])
		##### NORMALIZE!!!!!
		sim_heatmap_frac_of_Pplan[col] = (sim_heatmap[col] - np.nanmin(sim_heatmap[col])) / (np.nanmax(sim_heatmap[col]) - np.nanmin(sim_heatmap[col]))
		#num_nonzero_cells = len(np.where(sim_heatmap[col] > 0)[0])
		#sim_heatmap_frac_of_Pplan[col] = sim_heatmap_frac_of_Pplan[col] * num_nonzero_cells

		#single_heatmap_frac_of_Pplan[col] = single_heatmap[col] / np.nansum(single_heatmap[col]) 
		single_heatmap_frac_of_Pplan[col] = (single_heatmap[col] - np.nanmin(single_heatmap[col])) / (np.nanmax(single_heatmap[col]) - np.nanmin(single_heatmap[col]))
		#num_nonzero_cells = len(np.where(single_heatmap[col] > 0)[0])
		#single_heatmap_frac_of_Pplan[col] = single_heatmap_frac_of_Pplan[col] * num_nonzero_cells

		#multi_nonHL_heatmap_frac_of_Pplan[col] = multi_nonHL_heatmap[col] / np.nansum(multi_nonHL_heatmap[col])
		multi_nonHL_heatmap_frac_of_Pplan[col] = (multi_nonHL_heatmap[col] - np.nanmin(multi_nonHL_heatmap[col])) / (np.nanmax(multi_nonHL_heatmap[col]) - np.nanmin(multi_nonHL_heatmap[col]))
		#num_nonzero_cells = len(np.where(multi_nonHL_heatmap[col] > 0)[0])
		#multi_nonHL_heatmap_frac_of_Pplan[col] = multi_nonHL_heatmap_frac_of_Pplan[col] * num_nonzero_cells


		#multi_HL_heatmap_frac_of_Pplan[col] = multi_HL_heatmap[col] / np.nansum(multi_HL_heatmap[col])
		multi_HL_heatmap_frac_of_Pplan[col] = (multi_HL_heatmap[col] - np.nanmin(multi_HL_heatmap[col])) / (np.nanmax(multi_HL_heatmap[col]) - np.nanmin(multi_HL_heatmap[col]))
		#num_nonzero_cells = len(np.where(multi_HL_heatmap[col] > 0)[0])
		#multi_HL_heatmap_frac_of_Pplan[col] = multi_HL_heatmap_frac_of_Pplan[col] * num_nonzero_cells


	
	fig, ((ax1,ax2), (ax3, ax4)) = plt.subplots(nrows=2,ncols=2, sharex=True, sharey=True)
	#### upper left
	ax1.imshow(np.nan_to_num(sim_heatmap_frac_of_Pplan.T), origin='lower', aspect='auto', cmap='coolwarm')
	ax1.set_ylabel(r'$\log_{10} \, P_{\mathrm{TTV}}$ [epochs]')
	ax1.set_yticks(np.arange(0,19,1)[::4])
	ax1.set_yticklabels(np.around(np.log10(ycenters[::4]), 2))

	#### upper right
	ax2.imshow(np.nan_to_num(single_heatmap_frac_of_Pplan.T), origin='lower', aspect='auto', cmap='coolwarm')

	#### lower left
	ax3.imshow(np.nan_to_num(multi_nonHL_heatmap_frac_of_Pplan.T), origin='lower', aspect='auto', cmap='coolwarm')
	ax3.set_ylabel(r'$\log_{10} \, P_{\mathrm{TTV}}$ [epochs]')
	ax3.set_xlabel(r'$\log_{10} \, P_{\mathrm{P}}$ [days]')
	ax3.set_xticks(np.arange(0,19,1)[::4])
	ax3.set_xticklabels(np.around(np.log10(xcenters[::4]),2))
	ax3.set_yticks(np.arange(0,19,1)[::4])
	ax3.set_yticklabels(np.around(np.log10(ycenters[::4]),2))


	#### lower right
	ax4.imshow(np.nan_to_num(multi_HL_heatmap_frac_of_Pplan.T), origin='lower', aspect='auto', cmap='coolwarm')
	ax4.set_xlabel(r'$\log_{10} \, P_{\mathrm{P}}$ [days]')
	ax4.set_xticks(np.arange(0,19,1)[::4])
	ax4.set_xticklabels(np.around(np.log10(xcenters[::4]),2))
	plt.show()






	ax2.imshow(single_heatmap_frac_of_Pplan, origin='lower')


	ax3.imshow(multi_nonHL_heatmap_frac_of_Pplan, origin='lower')
	ax3.set_ylabel(r'$P_{\mathrm{TTV}}$ [epochs]')
	ax3.set_xlabel(r'$P_{\mathrm{P}}$ [days]')



	ax4.imshow(multi_HL_heatmap_frac_of_Pplan, origin='lower')
	ax4.set_xlabel(r'$P_{\mathrm{P}}$ [days]')
	plt.show()




















except:
	traceback.print_exc()



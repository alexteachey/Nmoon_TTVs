from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas
import os
import time
import traceback
from astropy.timeseries improt LombScargle
from astropy.io import ascii
from scipy.optimize import curve_fit

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

show_plots = input('Do you want to show plots (for debugging)? y/n: ')

OCfile = pandas.read_csv('/data/tethys/Documents/Software/MoonPy/Table3_O-C.csv')
KOIs = np.array(OCfile['KOI']).astype(str)
epochs = np.array(OCfile['n']).astype(int)
OCmin = np.array(OCfile['O-C_min']).astype(float)
OCmin_err = np.array(OCfile['O-C_err']).astype(float)

unique_KOIs = np.unique(KOIs)

cumkois = ascii.read('/data/tethys/Documents/Software/MoonPy/cumkois.txt')
kepois = np.array(cumkois['kepoi_name'])
kepoi_periods = np.array(cumkois['koi_period'])

kepoi_nums = []
for kepoi in kepois:
	kepoi_num = kepoi
	while (kepoi_num.startswith('K')) or (kepoi_num.startswith('0'):
		kepoi_num = kepoi_num[1:]
	kepoi_nums.append(kepoi_num)
kepoi_nums = np.array(kepoi_nums)
kepois = kepoi_nums


##### GENERATE LISTS 


for nkepoi, kepoi in enumerate(kepois):
	kepoi_period = kepoi_periods[nkepoi]
	KOI_idxs = np.where(KOIs == kepoi)[0]

	KOI_epochs, KOI_OCs, KOI_OCerrs = epochs[KOI_idxs], OCmin[KOI_idxs], OCmin_err[KOI_idxs]
	KOI_rms = np.nansqrt(np.nanmean(KOI_OCs**2))

	### run a Lomb-Scargle periodogram on this -- let the range be 2-500 epochs, 5000 logarithmically spaced bins.

	LSperiods = np.logspace(np.log10(2), np.logspace(500), 5000)
	LSfreqs = 1/LSperiods
	LSpowers = LombScargle(KOI_epochs, KOI_OCs, KOI_OCerrs).power(LSfreqs)
	peak_power_idx = np.nanargmax(LSpowers)
	peak_power_period = LSperiods[peak_power_idx]
	peak_power_freq = 1/peak_power_period


	if show_plots == 'y':
		plt.plot(LSperiods, LSpowers, facecolor='DodgerBlue', alpha=0.7, linewidth=2)
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
	popt, pcov = curve_fit(sinecurve, KOI_epochs, KOI_OCs, sigma=KOI_OCerrs, bounds=([0, -2*np.pi], [5*KOI_rms, 2*np.pi]))

	if show_plots == 'y':
		KOI_epochs_interp = np.linspace(np.nanmin(KOI_epochs), np.nanmax(KOI_epochs), 1000)
		KOI_TTV_interp = sinecuve(KOI_epochs_interp, *popt)

		plt.scatter(KOI_epochs, KOI_OCs, facecolor='LightCoral', edgecolor='k', alpha=0.7, zorder=2)
		plt.errorbar(KOI_epochs, KOI_OCs, yerr=KOI_OCerrs, ecolor='k', fmt='none', zorder=1)
		plt.plot(KOI_epochs_interp, KOI_TTV_interp, color='k', linestyle='--', linewidth=2)
		plt.xlabel("epoch")
		plt.ylabel('O - C [min]')
		plt.title('KOI-'+str(kepoi))
		plt.show()





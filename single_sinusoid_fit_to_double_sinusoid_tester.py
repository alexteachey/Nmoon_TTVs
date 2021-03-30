from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas
import os
import traceback
import time
from astropy.timeseries import LombScargle
from scipy.optimize import curve_fit

plt.rcParams["font.family"] = 'serif'


##### THIS CODE WILL TEST THE QUESTION: IS A SINGLE SINUSOID FIT *ALWAYS* BETTER THAN A FLAT LINE, WHEN MULTIPLE SIGNALS ARE PRESENT?
"""
Here's what we're gonna do. We're gonna generate a range of two-sinusoid curves, all fixed length, all fixed uncertainty. (LEAVE UNCERTAINTY OUT TO START!)
We're going to look at two (three?) factors... delta-amplitude, delta-phase, and delta-period.

THE PROBLEM THAT ALWAYS ARISES IS -- YOU'RE UNDERSAMPLING THESE!!!!!!!!




"""

try:

	def chisquare(data,model,error):
		return np.nansum(((data-model)**2 / (error**2)))

	def BIC(nparams,data,model,error):
		return nparams*np.log(len(data)) + chisquare(data,model,error)


	show_plots = input('Do you want to show plots? y/n: ')


	baseline = 1000 ### days
	Pplan = 50 #### days --- we'll get 20 observations
	times = np.arange(0,baseline+Pplan,Pplan)
	times_epochs = times / Pplan
	P1 = 4.8526  #### days -- making it weird so it doesn't line up cleanly with the planet's orbital period
	f1 = 1/P1
	w1 = 2*np.pi*f1
	P2 = np.linspace(0.2*P1, 5*P1, 20) #### range of values, from 1/5 the orbital period times 5 times the orbital period
	f2 = 1/P2
	w2 = 2*np.pi*f2
	ph1 = np.pi #### phase of the first moon
	ph2 = np.linspace(ph1-(2*np.pi), ph1+(2*np.pi), 20) #### range of values, phase of the second moon
	A1 = 10 #### minutes. Arbitrary.
	A2 = np.linspace(2,50,20) #### range of amplitudes

	#### note that 50x50x50 is 125000 runs! 20x20x20 is a more modest 8000 runs. Use that to start.

	coordinates_grid = np.zeros(shape=(20,20,20)) #### where we store the coordinates, i.e. the inputs
	deltaBICs_3Dgrid = np.zeros(shape=(20,20,20)) #### delta-amplitude, delta-period, delta-phase


	sine1 = A1 * np.sin(w1*times + ph1)

	for na2, amp2 in enumerate(A2):
		for nw2, omega2 in enumerate(w2):
			for nph2, phase2 in enumerate(ph2):
				print('A2, w2, ph2 = ', amp2, omega2, phase2)
				sine2 = amp2 * np.sin(omega2*times + phase2)
				combo_signal = sine1 + sine2
				noise_amplitude = 1 #### minute
				noise_array = np.linspace(noise_amplitude, noise_amplitude, len(combo_signal))
				noisy_combo_signal = np.random.normal(loc=combo_signal, scale=noise_array)

				delta_amplitude = A1 - amp2
				delta_omega = w1 - omega2
				delta_phase = ph1 - phase2

				#### now we're gonna run a LombScargle on this sucker, take the max period, and use that to generate a *single* moon fit to this
				periods_to_probe = np.logspace(np.log10(2),np.log10(100),1000)
				freqs_to_probe = 1 / periods_to_probe
				LSpowers = LombScargle(times_epochs, noisy_combo_signal, noise_array).power(freqs_to_probe)
				max_power_idx = np.nanargmax(LSpowers)
				max_power_period = periods_to_probe[max_power_idx]
				max_power_freq = 1 / max_power_period
				max_power_angfreq = 2*np.pi * max_power_freq


				#### use this period to do a curve fit

				def cfsine(tvals, amplitude, phase):
					return amplitude * np.sin(max_power_angfreq * tvals + phase)

				popt, pcov = curve_fit(f=cfsine, xdata=times_epochs, ydata=noisy_combo_signal, sigma=noise_array, bounds=[(0, -2*np.pi), (2*np.nanmax(np.abs(noisy_combo_signal)), 2*np.pi)])
				best_fit_curve = cfsine(times_epochs, *popt)
				supersample_times_epochs = np.linspace(np.nanmin(times_epochs), np.nanmax(times_epochs), 1000)



				BIC_flat = BIC(nparams=0, data=noisy_combo_signal, model=np.linspace(0,0,len(noisy_combo_signal)), error=noise_array)
				BIC_curve = BIC(nparams=2, data=noisy_combo_signal, model=best_fit_curve, error=noise_array)

				##### BIC_curve will have a larger value, all things being equal, by virtue of the klog(n) term. So you want to do BIC_curve - BIC_flat, and look for negative numbers.
				##### in this convention, if DeltaBIC is negative, it means BIC_curve is less than BIC_flat, which is what we want.

				deltaBIC = BIC_curve - BIC_flat 

				if show_plots == 'y':
					plt.scatter(times_epochs, noisy_combo_signal, facecolor='LightCoral', edgecolor='k', s=20, zorder=1)
					plt.errorbar(times_epochs, noisy_combo_signal, yerr=noise_array, ecolor='k', alpha=0.5, zorder=0, fmt='none')
					plt.plot(supersample_times_epochs, cfsine(supersample_times_epochs, *popt), c='r', linestyle='--')
					plt.xlabel('Epochs')
					plt.ylabel('O - C')
					plt.title(r'$\Delta$BIC = '+str(round(deltaBIC, 2)))
					plt.show()

				deltaBICs_3Dgrid[na2][nw2][nph2] = deltaBIC 
				print(' ')


except:
	traceback.print_exc()










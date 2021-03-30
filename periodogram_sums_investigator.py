from __future__ import division
import numpy as np
import pandas 
import matplotlib.pyplot as plt
import os
import traceback
import time
from astropy.timeseries import LombScargle


##### this will generate three sinusoids, a LS for each, superpose those sinusoids, and superpose the LS, and look at the difference between that one and the sum of the individuals.

try:

	ntrials = int(input('How many trials? '))
	use_noise = input('Use noise? y/n: ')

	nsignals = 3
	Pplan = np.random.randint(low=45,high=55)+np.random.random() ### days
	baseline = 1500 ### days
	transit_times = np.arange(0,baseline+Pplan,Pplan)
	random_idxs = np.sort(np.random.choice(np.arange(0,len(transit_times),1), int(0.95*len(transit_times))))
	transit_times = transit_times[random_idxs]
	Pmin = 2*Pplan
	Pmax = baseline
	sample_periods = np.logspace(np.log10(Pmin), np.log10(Pmax), 10000)
	sample_freqs = 1 / sample_periods

	for i in np.arange(0,ntrials,1):
		#### let the amplitudes all be the same
		random_periods = np.random.randint(low=2,high=20,size=nsignals) + np.random.random(size=nsignals)
		random_freqs = 1 / random_periods
		random_angfreqs = 2*np.pi * random_freqs
		random_phases = np.random.choice(np.linspace(-2*np.pi, 2*np.pi, 1000), size=nsignals)
		amplitude = 5 ### minutes

		signals = np.zeros(shape=(len(random_angfreqs), len(transit_times)))
		sidx = 0
		for raf, rph in zip(random_angfreqs, random_phases):
			signal = amplitude * np.sin(raf * transit_times + rph)
			signals[sidx] = signal
			sidx += 1

		##### LEAVE OUT THE NOISE TO START WITH!!!!!
		noise = 0.5
		noise_array = np.linspace(noise, noise, len(transit_times))
		noisy_signals = np.random.normal(loc=signals, scale=noise_array)

		LS_stack = np.zeros(shape=(nsignals, len(sample_freqs)))

		if use_noise == 'y':
			signal_input = noisy_signals
		else:
			signal_input = signals 

		for nsignal, signal in enumerate(signal_input):
			#LSpowers = LombScargle(transit_times, signal, noise_array).power(sample_freqs)
			if use_noise == 'y':
				LSpowers = LombScargle(transit_times, signal, noise_array).power(sample_freqs)
			else:
				LSpowers = LombScargle(transit_times, signal).power(sample_freqs)

			LS_stack[nsignal] = LSpowers

		#summed_signal = np.sum(noisy_signals, axis=0)
		if use_noise == 'y':
			summed_signal = np.random.normal(loc=np.sum(signals, axis=0), scale=noise_array)
		else:
			summed_signal = np.random.normal(loc=np.sum(signals, axis=0))

		summed_LSpowers = np.sum(LS_stack, axis=0)

		### now run a periodogram on summed_signal, and see how it compares to summed_LSpowers
		if use_noise == 'y':
			summed_signal_LS = LombScargle(transit_times, summed_signal, noise_array).power(sample_freqs)
		else:
			summed_signal_LS = LombScargle(transit_times, summed_signal).power(sample_freqs)

		fig, ax = plt.subplots(4, 2, figsize=(8,8))
		if use_noise == 'y':
			ax[0][0].scatter(transit_times, noisy_signals[0], facecolor='LightCoral', edgecolor='k', s=20)
		else:
			ax[0][0].scatter(transit_times, signals[0], facecolor='LightCoral', edgecolor='k', s=20)
		ax[0][1].plot(sample_periods, LS_stack[0], c='r', alpha=0.5, label='LS1')
		ax[0][1].legend()
		ax[0][1].set_xscale('log')

		if use_noise == 'y':
			ax[1][0].scatter(transit_times, noisy_signals[1], facecolor='LightCoral', edgecolor='k', s=20)
		else:
			ax[1][0].scatter(transit_times, signals[1], facecolor='LightCoral', edgecolor='k', s=20)

		ax[1][1].plot(sample_periods, LS_stack[1], c='r', alpha=0.5, label='LS2')
		ax[1][1].set_xscale('log')	
		ax[1][1].legend()

		if use_noise == 'y':
			ax[2][0].scatter(transit_times, noisy_signals[2], facecolor='LightCoral', edgecolor='k', s=20)
		else:
			ax[2][0].scatter(transit_times, signals[2], facecolor='LightCoral', edgecolor='k', s=20)

		ax[2][1].plot(sample_periods, LS_stack[2], c='r', alpha=0.5, label='LS3')
		ax[2][1].set_xscale('log')
		ax[2][1].legend()

		ax[3][0].scatter(transit_times, summed_signal, facecolor='LightCoral', edgecolor='k', s=20)
		ax[3][1].plot(sample_periods, summed_LSpowers, c='k', alpha=0.5, label='LS1+LS2+LS3')
		ax[3][1].plot(sample_periods, summed_signal_LS, c='r', alpha=0.5, label='LS(S1+S2+S3)') 
		ax[3][1].set_xscale('log')
		ax[3][1].legend()

		plt.subplots_adjust(left=0.125, bottom=0.09, right=0.9, top=0.95, wspace=0.2, hspace=0.05)
		
		plt.show()

except:
	traceback.print_exc()

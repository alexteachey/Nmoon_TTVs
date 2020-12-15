from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas
from astropy.timeseries import LombScargle
from scipy.interpolate import interp1d, interp2d
import traceback
from astropy.constants import G, R_sun, M_sun, M_jup, R_jup
import pickle
import os


projectdir = '/data/tethys/Documents/Projects/NMoon_TTVs'
dictdir = projectdir+'/sinusim_dictionaries'
waterfalldir = projectdir+'/sinusim_waterfalls'

"""
#### this script is a playground. The idea is to see what happens to a multi-sinusoidal signal when you selectively remove the undersampled observations.

Here's the thinking: suppose we have a multiple-moon signal, comprised of a series of sinusoids (the real signal will be a bit more complicated).

For simplicity, let's assume a three moon system, with a 1:2:4 resonance. The phases will be randomized. So you have P1, P2 = 2P1, and P3 = 4P1 = 2P2.

The amplitude of a TTV is proportional to mass of the moon and the semimajor axis. The semimajor axis is related to the period as P^2 / a^3 = (4pi^2) / (G(M+m))

Now this will produce a signal... then we will SAMPLE this signal at the period of the planet's orbit. Then we run a LS on that TTV to get a peak period.

NOW WHAT HAPPENS IF WE START DROPPING OBSERVATIONS? WILL THE PEAK BE THE SAME? WILL IT CHANGE? WHAT MIGHT IT TELL US ABOUT THE MOONS PRESENT?!?!

This is exploratory, just see what happens to the periodograms.

"""

def Tdur(Pplan_days):
	Pplan_seconds = Pplan_days * 24 * 60 * 60
	aplan_meters = ((Pplan_seconds**2 * G.value * (M_sun.value + M_jup.value)) / (4*np.pi**2))**(1/3)
	first_term = (Pplan_seconds / 2*np.pi)
	second_term_arg_num = R_sun.value
	second_term_arg_denom = 2 * aplan_meters
	second_term_arg = second_term_arg_num / second_term_arg_denom
	second_term = np.arctan(second_term_arg)

	Tdur_seconds = first_term * second_term
	Tdur_days = Tdur_seconds / (60 * 60 * 24)
	return Tdur_days 





show_plots = input('Do you want to show plots? y/n: ')
run_obs_stripping_test = input('Do you want to run the observation stripping experiment? y/n: ')


peak_LS_periods = []
peak_LS_periods_epochs = []



### make 100,000 examples!
nmade = len(os.listdir(waterfalldir))
ntomake = int(input("You have generated "+str(nmade)+' sims so far. How many total do you want? '))

try:
	#planet_periods = np.arange(50,1500,1)
	baseline = 3650 #days, = 10 years

	#for npplan, pplan in enumerate(planet_periods):
	while nmade < ntomake:
		print('sinusim: ', nmade)

		system_dict = {}

		pplan = np.random.choice(np.arange(20,500,1))+np.random.random()
		ntransits = (baseline / pplan).astype(int) #### longer the planet period, the fewer the transits.

		system_dict['Pplan'] = pplan
		system_dict['ntransits'] = ntransits


		nmoons = np.random.randint(1,6) ### 1,2,3,4 or 5 moons
		#moon_amplitudes = np.array([2,4,8])
		moon_phases = np.random.choice(np.linspace(0,2*np.pi,1000), nmoons)
		
		moon_periods_days = [np.random.randint(1,4)+np.random.random()] ### days
		for i in np.arange(1,nmoons,1):
			moon_periods_days.append(2.05*moon_periods_days[-1])
		moon_periods_days = np.array(moon_periods_days)
		moon_periods_seconds = moon_periods_days * 24 * 60 * 60

		system_dict['Nmoons'] = nmoons 
		system_dict['Pmoons'] = moon_periods_days


		### calculate a moon semimajor axis, assume Mp = M_jup
		mass_ratio = 1e-2 ### will determine the physical swing of the planet around the barycenter.
		moon_smas_meters = ((moon_periods_seconds**2 * G.value * (M_jup.value + mass_ratio*M_jup.value)) / (4*np.pi**2))**(1/3)
		#### in the limit of equal mass, the barycenter is halfway between. where satellite = 0, no barycenter offset.
		distances_from_barycenter_meters = moon_smas_meters - (M_jup.value / (M_jup.value + mass_ratio*M_jup.value) * moon_smas_meters) 
		### ^ the above are PHYSICAL AMPLITUDES. YOU NEED TO CONVERT THEM TO *TIME* AMPLITUDES.
		Tdur_days = Tdur(pplan)
		#Tdur_days_sigma = 0.04*Tdur_days
		Tdur_days_sigma = 30 / (60 * 24) ### 30 minutes in days

		vx_mperday = R_sun.value / Tdur_days
		time_amplitudes_days = distances_from_barycenter_meters / vx_mperday ### moon time amplitudes.

		moon_TTV_amplitudes_minutes = time_amplitudes_days * 24 * 60
		moon_freqs_perday = 1 / moon_periods_days
		moon_angfreqs_perday = 2*np.pi * moon_freqs_perday




		#### generate the moon signals

		moon_times = np.linspace(0,pplan*ntransits,10000) ### 100,000 samples over the course of 20 planet transits

		moon_signals = np.zeros(shape=(nmoons,len(moon_times)))


		"""
		### we're already in time units, don't need to do this conversion.
		vx = R_sun.value / Tdur_days #### meters / days

		#### uncertainty in space will be just uncertainty in Tdur, scaled by vx
		x_sigma = Tdur_days_sigma
		"""

		for i in np.arange(0,nmoons,1):
			moon_signals[i] = time_amplitudes_days[i] * np.sin(moon_angfreqs_perday[i]*moon_times + moon_phases[i]) 

		#### sum them up!
		summed_moon_signal = np.nansum(moon_signals, axis=0) 

		summed_moon_signal_interpolator = interp1d(moon_times, summed_moon_signal)


		#### now grab the moon signal at the interpolated times

		sample_times = np.arange(0,pplan*ntransits,pplan) #### sample once per planet period.
		sample_summed_moon_signal = summed_moon_signal_interpolator(sample_times)
		noisy_sample_summed_moon_signal = np.random.normal(loc=sample_summed_moon_signal, scale=Tdur_days_sigma)

		#### let's plot it to make sure we're doing this right! -- SEEMS TO WORK WELL.
		if show_plots == 'y':
			plt.plot(moon_times, summed_moon_signal, c='k', alpha=0.5, zorder=0)
			plt.scatter(sample_times, noisy_sample_summed_moon_signal, s=30, facecolor='red', edgecolor='k', zorder=2)
			plt.errorbar(sample_times, noisy_sample_summed_moon_signal, yerr=Tdur_days_sigma, fmt='none', ecolor='k', zorder=1)
			plt.xlabel('Time [days]')
			plt.ylabel('O - C [days]')
			plt.show()
		

		if run_obs_stripping_test == 'y':
			ntrials = len(sample_times) - 5 ### leave five samples at the end.
		else:
			ntrials = 1

		trial_peak_periods = []
		trial_peak_periods_epochs = []
		keep_idxs = np.arange(0,len(sample_times),1)


		np.random.seed(42)
		for trial in np.arange(0,ntrials,1):

			#### you will remove N = trial observations at random and see what happens to the periodogram!
			#### alternatively, you may keep N = (len(sample_times) - trial) indices.
			#### call these the "keep_indices" keep_idxs

			ntokeep = len(sample_times) - trial
			#keep_idxs = np.random.choice(np.arange(0,len(sample_times),1), ntokeep)
			keep_idxs = np.delete(keep_idxs, np.random.randint(0,len(keep_idxs),1))

			LS_min_period = 2*pplan
			LS_max_period = 2*baseline 

			LS_freqs_to_test = np.logspace(np.log10(1/LS_max_period), np.log10(1/LS_min_period), 1000)
			LS_periods_to_test = 1/LS_freqs_to_test
			LS_periods_to_test_in_epochs = LS_periods_to_test / pplan

			LS_powers = LombScargle(sample_times[keep_idxs], noisy_sample_summed_moon_signal[keep_idxs], Tdur_days_sigma).power(LS_freqs_to_test)

			if trial == 0:
				LS_power_stack = LS_powers
			else:
				LS_power_stack = np.vstack((LS_power_stack, LS_powers))

			LS_peak_power_period = LS_periods_to_test[np.argmax(LS_powers)]
			LS_peak_power_period_in_epochs = LS_peak_power_period / pplan 

			peak_LS_periods.append(LS_peak_power_period)
			peak_LS_periods_epochs.append(LS_peak_power_period_in_epochs)

			trial_peak_periods.append(LS_peak_power_period)
			trial_peak_periods_epochs.append(LS_peak_power_period_in_epochs)

			"""
			if show_plots == 'y':
				
				plt.plot(LS_periods_to_test_in_epochs, LS_powers)
				plt.xscale('log')
				plt.xlabel('Period [epochs]')
				plt.show()	
			"""

		if run_obs_stripping_test == 'y':
			if show_plots == 'y':
				plt.scatter(np.arange(0,ntrials,1), trial_peak_periods_epochs, s=30, facecolor='red', edgecolor='k')
				plt.plot(np.arange(0,ntrials,1), trial_peak_periods_epochs, c='k', alpha=0.5)
				plt.xlabel('# of obs removed')
				plt.ylabel(r'$P_{TTV}$ [epochs]')
				plt.yscale('log')
				plt.title('Planet Period = '+str(pplan)+'days, # moons = '+str(nmoons))
				plt.show()

			#### plot a histogram!
			if show_plots == 'y':
				histbins = np.arange(1.5,ntransits+0.5,1)
				n, bins, edges = plt.hist(trial_peak_periods_epochs, facecolor='NavajoWhite', edgecolor='k', bins=histbins)
				plt.xlabel('peak period [epochs]')
				plt.title(r'$P_P = $'+str(pplan)+' days, # moons = '+str(nmoons))
				plt.show()

			#### plot mean power at every period, and plot the percentiles!
			median_LS_powers = np.nanmedian(LS_power_stack, axis=0)
			low2sig = np.percentile(LS_power_stack, 2.5, axis=0)
			up2sig = np.percentile(LS_power_stack, 97.5, axis=0)

			#plt.plot(LS_periods_to_test_in_epochs, median_LS_powers, c='r', linewidth=2, zorder=2)
			#plt.plot(LS_periods_to_test_in_epochs, low2sig, c='k', linestyle='--', zorder=0)
			#plt.plot(LS_periods_to_test_in_epochs, up2sig, c='k', linestyle='--', zorder=0)
			#plt.fill_between(LS_periods_to_test_in_epochs, low2sig, up2sig, color='DodgerBlue', zorder=1, alpha=0.5)
			if show_plots == 'y':
				for i in np.arange(0,LS_power_stack.shape[0],1):
					plt.plot(LS_periods_to_test_in_epochs, LS_power_stack[i], c='k', alpha=0.5, zorder=2)
				plt.xlabel(r'$P_{TTV}$ [epochs]')
				plt.xscale('log')
				plt.title('Planet Period = '+str(pplan)+'days, # moons = '+str(nmoons))
				plt.show()

			#### WATERFALL PLOT! SPECTRAL EVOLUTION AS YOU STRIP AWAY OBSERVATIONS!!!!!
			LS_power_stack_interpolator = interp2d(LS_periods_to_test_in_epochs, np.linspace(0,1,ntrials), LS_power_stack)
			LS_power_interped = LS_power_stack_interpolator(LS_periods_to_test_in_epochs, np.linspace(0,1,100)) #### shape will be 1000 x 100 (1000 x 1000 is insane!)

			if show_plots == 'y':
				plt.imshow(LS_power_interped, origin='lower', cmap='cividis')
				plt.xlabel('Period [epochs]')
				plt.ylabel('# obs removed')
				plt.show()

			waterfall = LS_power_interped


		#### save this shit!
		with open(dictdir+'/sinusim'+str(nmade)+'dictionary.pkl', 'wb') as handle:
			pickle.dump(system_dict, handle)

		np.save(waterfalldir+'/sinusim'+str(nmade)+'_waterfall.npy', waterfall)

		nmade += 1




	if show_plots == 'y':
		histbins = np.arange(1.5,2*ntransits+0.5,1)
		n, bins, edges = plt.hist(peak_LS_periods_epochs, bins=histbins, facecolor='NavajoWhite', edgecolor='k')
		plt.xlabel(r'$P_{TTV}$ (Epochs)')
		plt.ylabel('#')
		plt.show()



except:
	traceback.print_exc()







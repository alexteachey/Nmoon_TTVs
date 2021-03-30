from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas
import os
import time
import traceback
from astropy.timeseries import LombScargle
import matplotlib.cm as cm
from scipy.optimize import curve_fit
import pickle
from astropy.constants import M_sun, R_sun, M_jup, R_jup, M_earth, R_earth, G 

plt.rcParams["font.family"] = 'serif'


#### THIS CODE WILL ATTEMPT TO CONNECT THE PERIODOGRAM POWER AMPLITUDE TO THE SINUSOID AMPLITUDE.


def gaussian(xvals, mu, sigma):
	first_term = 1 / (sigma * np.sqrt(2*np.pi))
	exponent_arg = -0.5 * ((xvals - mu) / sigma)**2
	return first_term * np.exp(exponent_arg)

def freesine(xvals, angfreq, amplitude, phase):
	#### supply the angular frequency
	return amplitude * np.sin((angfreq * xvals) + phase)


def Tdur_hours_calc(Pplan_days, rstar=R_sun.value, mstar=M_sun.value, mplan=M_jup.value):
	Pplan_hours = Pplan_days * 24
	Pplan_seconds = Pplan_days * 24 * 60 * 60
	arctan_arg_num = ((Pplan_seconds**2 * G.value * (mstar + mplan)) / (4*np.pi**2))**(1/3)
	arctan_arg = arctan_arg_num / rstar
	theta = np.arctan(arctan_arg)
	Tdur_hours = (theta / (2*np.pi)) * Pplan_hours 
	return Tdur_hours

def TTVampcalc(Pplan_days, Pmoon_days, mass_ratio, rstar=R_sun.value, mstar=M_sun.value, mplan=M_jup.value):
	#### assumes star is solar and planet is jupiter.
	Pplan_hours = Pplan_days * 24
	Pplan_seconds = Pplan_days * 24 * 60 * 60
	arctan_arg_num = ((Pplan_seconds**2 * G.value * (mstar + mplan)) / (4*np.pi**2))**(1/3)
	arctan_arg = arctan_arg_num / rstar
	theta = np.arctan(arctan_arg)

	Tdur_hours = (theta / (2*np.pi)) * Pplan_hours 

	vx_meters_per_hour = (2*rstar) / Tdur_hours

	Pmoon_hours = Pmoon_days * 24 
	Pmoon_seconds = Pmoon_days * 24 * 60 * 60 
	amoon_meters = ((Pmoon_seconds**2 * G.value * (mplan + (mplan*mass_ratio))) / (4*np.pi**2))**(1/3)

	#### if mass_ratio == 0, xdisp = 0.
	#### if mass ratio == 1, xdisp = 0.5*amoon_meters.
	xdisp_meters = mass_ratio * (0.5*amoon_meters)
	deltaT_hours = xdisp_meters / vx_meters_per_hour 
	deltaT_minutes = deltaT_hours * 60 
	deltaT_seconds = deltaT_minutes * 60
	return deltaT_seconds 



num_iterations = 6


show_plots = input('Do you want to show plots? y/n: ')
load_real_TTVs = input('Load real TTVs? y/n: ')
if load_real_TTVs == 'y':
	ntrials = 50000
	sim_prefix = input("What is the simulation prefix? ")
	if (len(sim_prefix) != 0) and (sim_prefix[-1] != '_'):
		sim_prefix = sim_prefix+'_'
else:
	ntrials = int(input("How many trials do you want to run? "))

write_to_file = input('Do you want to write to file? y/n: ')


try:






	pmin, pmax = 2, 500 #### EPOCHS
	test_frequencies = np.logspace(np.log10(1/pmax), np.log10(1/pmin), 10000) ### inverse epochs
	amplitudes = []
	frequencies = []
	periods = []
	nsignals = []
	niterations = []

	##### choose some points at random
	#baseline = np.random.choice(a=baseline, size=int(0.8*len(baseline)))

	trial_dict = {}


	for trial in np.arange(0,ntrials,1): #### a trial is a single system.

		try:

			print('trial #: ', trial)
			trial_dict[trial] = {}

			##### select a number of sinusoids to generate, with random amplitudes, frequencies, and phases
			##### then we want to see how the LS looks

			trial_amps = []
			trial_freqs = []
			trial_phases = []
			trial_periods = []
			trial_FAPs = []
			

			if load_real_TTVs != 'y':

				#### CHOOSE the number of moons, the orbital period of the planet, and the orbital period of each moon.
				nmoons = np.random.randint(low=1, high=6) #### between 1 and 5 moons
				Pmoons = np.random.randint(low=1,high=20,size=nmoons)+np.random.random(size=nmoons)
				Pplan = np.random.randint(low=10,high=365)+np.random.random()
				baseline_days = np.arange(0,3650+Pplan,Pplan) #### 10 year baseline, in days
				baseline_epochs = baseline_days / Pplan
				baseline_separation = baseline_epochs[1] - baseline_epochs[0]
				baseline_frequency_sep = 1/baseline_separation 
				#moon_mass_ratios = np.random.randint(low=-5,high=-2,size=nmoons)-np.random.random(size=nmoons)
				moon_mass_ratios = np.random.randint(low=-4,high=-2)-np.random.random(size=nmoons) #### keep them all in the same ballpark!
				moon_mass_ratios = 10**moon_mass_ratios

				#### DETERMINE WHETHER IT SHOULD BE INCLUDED, BASED ON GEOMETRIC TRANSIT PROBABILITY
				Pplan_hours = Pplan * 24
				Pplan_seconds = Pplan * 24 * 60 * 60
				arctan_arg_num = ((Pplan_seconds**2 * G.value * (M_sun.value + M_jup.value)) / (4*np.pi**2))**(1/3)
				arctan_arg = arctan_arg_num / R_sun.value
				theta = np.arctan(arctan_arg)
				transit_probability = theta / np.pi #### theta is the angle subtended by half the stellar disk, so 2theta is the whole transit opportunity, div by 2pi, 2s cancel out.

				total_sinusoid = np.linspace(0,0,len(baseline_epochs))

				rand_nsignals = nmoons ### for backwards compatibility
				colors = cm.get_cmap('viridis')(np.linspace(0,1,5))
				color_idxs = np.linspace(0,1,5)		

				nsignal_range = np.arange(0,rand_nsignals,1)

				for nsig in nsignal_range:
					#### compute the amplitude, in seconds, of the moon signal
					Pmoon_days = Pmoons[nsig]
					moon_mass_ratio = moon_mass_ratios[nsig]
					moon_TTVamp_seconds = TTVampcalc(Pplan_days=Pplan, Pmoon_days=Pmoon_days, mass_ratio=moon_mass_ratio)
					random_amp = moon_TTVamp_seconds #### for backward compatibility
					Pmoon_epochs = Pmoon_days / Pplan
					random_period = Pmoon_epochs #### for backward compatibility

					random_freq = 1 / random_period
					random_angfreq = 2*np.pi*random_freq
					random_phase = np.random.choice(np.linspace(-2*np.pi,2*np.pi, 10000))
					random_sinusoid = freesine(xvals=baseline_epochs, angfreq=random_angfreq, amplitude=random_amp, phase=random_phase)
					
					total_sinusoid += random_sinusoid 
					trial_amps.append(random_amp)
					trial_freqs.append(random_freq)
					trial_phases.append(random_phase)
					trial_periods.append(random_period)

				print('trial_amps: ', trial_amps)
				print('trial_freqs: ', trial_freqs)
				print("trial_periods: ", trial_periods)
				print('trial_phases: ', trial_phases)

				#### noise it up! 
				#sinusoid_sigma = 0.1*np.nanmax(total_sinusoid)
				nobs_per_transit = 2 * Tdur_hours_calc(Pplan)
				transit_depth_ppm = 1e6 * (R_jup.value / R_sun.value)**2
				transit_SNR = (transit_depth_ppm * np.sqrt(nobs_per_transit)) / 350 #### 350 ppm
				timing_uncertainty_minutes = 100 / transit_SNR
				timing_uncertainty_seconds = timing_uncertainty_minutes * 60
				sinusoid_sigma = timing_uncertainty_seconds ### for backward compatibility
				total_sinusoid = np.random.normal(loc=total_sinusoid, scale=timing_uncertainty_seconds)
				#### have to re-center the sinusoid on zero! 
				total_sinusoid = total_sinusoid - np.nanmedian(total_sinusoid)


				trial_dict[trial]['nsignals'] = rand_nsignals
				trial_dict[trial]['amps'] = np.array(trial_amps)
				trial_dict[trial]['periods'] = np.array(trial_periods)
				trial_dict[trial]['phases'] = np.array(trial_phases)
				trial_dict[trial]['freqs'] = np.array(trial_freqs)
				trial_dict[trial]['angfreqs'] = 2*np.pi * np.array(trial_freqs)

				random_idxs = np.random.randint(low=0, high=len(baseline_epochs), size=int(0.95*len(baseline_epochs)))




			elif load_real_TTVs == 'y':
				modeldictdir = '/run/media/amteachey/Auddy_Akiti/Teachey/Nmoon_TTVs/'+sim_prefix+'sim_model_settings'
				if 'RUN4' in sim_prefix:
					TTVdir = '/run/media/amteachey/Auddy_Akiti/Teachey/Nmoon_TTVs/'+sim_prefix+'FIXED_sim_TTVs'
				else:
					TTVdir = '/run/media/amteachey/Auddy_Akiti/Teachey/Nmoon_TTVs/'+sim_prefix+'sim_TTVs'					
				TTVfile = pandas.read_csv(TTVdir+'/TTVsim'+str(trial)+'_TTVs.csv')
				baseline_epochs = np.array(TTVfile['epoch']).astype(float)
				total_sinusoid = np.array(TTVfile['TTVob']).astype(float)
				sinusoid_sigma = np.nanmedian(np.array(TTVfile['timing_error']).astype(float))
				random_idxs = np.arange(0,len(baseline_epochs),1) ### don't omit any!
				modeldict = pickle.load(open(modeldictdir+'/TTVsim'+str(trial)+'_system_dictionary.pkl', 'rb'))
				nmoons = len(modeldict.keys())-1 #### if you have 3 moons, you have ['Planet', 'I', 'II', 'III'] = 4 keys.
				rand_nsignals = nmoons
				nsignal_range = np.arange(0,rand_nsignals,1)

				Pplan_seconds = modeldict['Planet']['P']
				if 'RUN4' in sim_prefix:
					arctan_arg_num = ((Pplan_seconds**2 * G.value * (M_sun.value + M_jup.value)) / (4*np.pi**2))**(1/3)
					arctan_arg = arctan_arg_num / R_sun.value
					theta = np.arctan(arctan_arg)
				else:
					Rstar_meters = modeldict['Planet']['Rstar']
					Rstar_Rsol = Rstar_meters / R_sun.value
					Mstar_Msol = Rstar_Rsol**(1.136) 
					Mstar_kg = Mstar_Msol * M_sun.value
					Mplan_kg = modeldict['Planet']['m']
					arctan_arg_num = ((Pplan_seconds**2 * G.value * (Mstar_kg + Mplan_kg)) / (4*np.pi**2))**(1/3)					
					arctan_arg = arctan_arg_num / Rstar_meters
					theta = np.arctan(arctan_arg)			

				transit_probability = theta / np.pi #### theta is half the stellar radius, 2theta is the whole disk, div by 2pi, 2s cancel.


			random_transit_probability_draw = np.random.random()
			if transit_probability <random_transit_probability_draw:
				#### SKIP THIS ONE!
				print('DID NOT TRANSIT. SKIPPING.')
				continue



			#### run the LombScargle
			LS = LombScargle(baseline_epochs[random_idxs], total_sinusoid[random_idxs])
			LSpowers = LS.power(test_frequencies)
			#test_frequencies, LSpowers = LombScargle(baseline_epochs, total_sinusoid).autopower(nyquist_factor=2)
			LSperiods = 1 / test_frequencies
			LS_FAP = LS.false_alarm_probability(LSpowers.max())
			trial_FAPs.append(LS_FAP)
			print('ORIGINAL FAP: ', LS_FAP)

			LSpowers = LSpowers * np.nanmax(np.abs(total_sinusoid))

			if show_plots == 'y':
				fig, (ax1, ax2) = plt.subplots(2, figsize=(6,8))

				#ax1.plot(baseline_epochs, total_sinusoid, c='k', linewidth=2, alpha=0.5)
				ax1.scatter(baseline_epochs[random_idxs], total_sinusoid[random_idxs], facecolor='LightCoral', edgecolor='k', s=20, zorder=1)
				ax1.errorbar(baseline_epochs[random_idxs], total_sinusoid[random_idxs], yerr=sinusoid_sigma, ecolor='k', zorder=0, fmt='none')
				ax2.plot(LSperiods, LSpowers, c='k', linewidth=2, alpha=0.5)
				for n,ta,tf,tph,tp in zip(nsignal_range,trial_amps, trial_freqs, trial_phases, trial_periods):
					trial_sine = freesine(xvals=baseline_epochs, angfreq=2*np.pi*tf, amplitude=ta, phase=tph)
					ax1.plot(baseline_epochs, trial_sine, color=colors[n], alpha=0.7)
					#ax2.plot(np.linspace(tp,tp,100), np.linspace(0,1.05*np.nanmax(LSpowers),100), color=colors[n], alpha=0.7)

				#ax2.plot(test_frequencies, LSpowers, c='DodgerBlue', linewidth=2)
				ax2.set_xscale('log')
				ax1.set_title("N = "+str(rand_nsignals)+', FAP = '+str(round(LS_FAP, 2)))
				plt.show()
			print(' ')
			print(' ')




			last_FAP = 0
			iteration = 0
			print('RUNNING REMOVAL TRIAL')
			total_removal_test_sinusoid = total_sinusoid
			
			#for nsig in nsignal_range:
			#for nsig in np.arange(0,5,1):
			for nsig in np.arange(0,num_iterations,1): #### TEST 10 ITERATIONS!!!!!
			#while last_FAP < 0.01:
				iteration += 1
				##### ITERATE ON THIS UNTIL FALSE ALARM PROBABILITY CROSSES 0.01.

				#### SEE ABOUT PEAK REMOVAL, ONE AT A TIME
				#peak_power_idx = np.argmax(LSpowers[100:-100])
				peak_power_idx = np.argmax(LSpowers)
				peak_power = LSpowers[peak_power_idx]
				peak_power_frequency = test_frequencies[peak_power_idx]
				peak_power_angfreq = 2*np.pi*peak_power_frequency
				#### DO A CURVE-FIT, with boundary on the amplitude between 0.8 and 1 peak power

				
				#### FIT A GAUSSIAN TO THE PEAK!!!!
				
				if (peak_power_idx > 20) and (peak_power_idx < len(test_frequencies)-21):
					local_idxs = np.arange(peak_power_idx-20,peak_power_idx+21,1)
				else:
					if peak_power_idx < 20:
						local_idxs = np.arange(0,peak_power_idx+21,1)
					elif peak_power_idx > len(test_frequencies)-21:
						local_idxs = np.arange(peak_power_idx-20,len(test_frequencies),1)

				local_xvals = test_frequencies[local_idxs]
				local_LSpowers = LSpowers[local_idxs]
				
				"""
				gauss_popt, gauss_pcov = curve_fit(f=gaussian, xdata=local_xvals - np.nanmedian(local_xvals), ydata=local_LSpowers)
				gauss_mu, gauss_sig = gauss_popt
				gauss_mu = gauss_mu + np.nanmedian(local_xvals)
				gauss_amp = np.nanmax(gaussian(local_xvals, *gauss_popt))
				print('gauss_amp: ', gauss_amp)
				print('gauss_mu: ', gauss_mu)
				print('gauss_sig: ', gauss_sig)
				gauss_residual = LSpowers[local_idxs] - gaussian(local_xvals, *gauss_popt)
				gauss_power_diff = np.abs(peak_power - gauss_amp) 
				"""

				print('peak power = ', peak_power)
				print('peak_power_idx = ', peak_power_idx)
				print('peak_power_period = ', LSperiods[peak_power_idx])
				print('fixedsine period: ', 1/(peak_power_angfreq/(2*np.pi)))

				def fixedsine(xvals, amplitude, phase):
					#### angfreq must be defined outside the function
					return amplitude * np.sin((peak_power_angfreq * xvals) + phase) #Asin(wt + phase)

				popt, pcov = curve_fit(f=fixedsine, xdata=baseline_epochs[random_idxs], ydata=total_removal_test_sinusoid[random_idxs], bounds=([0.1*peak_power,-2*np.pi],[peak_power,2*np.pi]))
				#popt, pcov = curve_fit(f=freesine, xdata=baseline_epochs, ydata=total_removal_test_sinusoid, bounds=([2*np.pi*(gauss_mu - gauss_sig), (gauss_amp - gauss_power_diff), -2*np.pi], [2*np.pi*(gauss_mu + gauss_sig), (gauss_amp + gauss_power_diff), 2*np.pi])) 
				#popt, pcov = curve_fit(f=freesine, xdata=baseline_epochs, ydata=total_removal_test_sinusoid, bounds=(2*np.pi*))
				print('popt: ', popt)		

				if show_plots == 'y':
					fig, (ax1, ax2) = plt.subplots(2, figsize=(6,8))
					ax1.set_title('REMOVING SINUSOID (iteration '+str(iteration)+', N ='+str(rand_nsignals)+')')
					#ax1.plot(baseline_epochs[random_idxs], total_removal_test_sinusoid[random_idxs], c='k', linewidth=2, alpha=0.5)
					ax1.scatter(baseline_epochs[random_idxs], total_removal_test_sinusoid[random_idxs], facecolor='LightCoral', edgecolor='k', s=20, zorder=1)
					ax1.errorbar(baseline_epochs[random_idxs], total_removal_test_sinusoid[random_idxs], yerr=sinusoid_sigma, ecolor='k', fmt='none', zorder=0)				
					ax1.plot(baseline_epochs, fixedsine(baseline_epochs, *popt), c='r', linestyle='--')		
					#print('called fixedsine with peak_power_angfreq: ', peak_power_angfreq)
					#print("That's f = "+str(peak_power_angfreq / (2*np.pi)))
					#print('Or P = '+str(1 / (peak_power_angfreq / (2*np.pi))))
					#ax1.plot(baseline_epochs, freesine(baseline_epochs, *popt), c='r', linestyle='--')
					ax2.plot(LSperiods, LSpowers, c='k', linewidth=2, alpha=0.5)
					ax2.plot(LSperiods[local_idxs], LSpowers[local_idxs], c='r', linestyle='--')
					#for n,ta,tf,tph,tp in zip(nsignal_range,trial_amps, trial_freqs, trial_phases, trial_periods):
					#	ax2.plot(np.linspace(tp,tp,100), np.linspace(0,1.05*np.nanmax(LSpowers),100), color=colors[n], alpha=0.7)
					ax2.set_xscale('log')
					plt.show()




				#### now subtract off this curve you just produced to
				last_LSpowers = LSpowers
				total_removal_test_sinusoid = total_removal_test_sinusoid - fixedsine(baseline_epochs, *popt)
				#### RE-CENTER IT
				total_removal_test_sinusoid = total_removal_test_sinusoid - np.nanmedian(total_removal_test_sinusoid)

				LS = LombScargle(baseline_epochs[random_idxs], total_removal_test_sinusoid[random_idxs])
				LSpowers = LS.power(test_frequencies)
				
				LS_FAP = LS.false_alarm_probability(LSpowers.max())
				trial_FAPs.append(LS_FAP)
				print('NEW FAP: ', LS_FAP)

				#total_removal_test_sinusoid = total_removal_test_sinusoid - freesine(baseline_epochs, *popt)
				#test_frequencies, LSpowers = LombScargle(baseline_epochs, total_removal_test_sinusoid).autopower(nyquist_factor=2)
				LSperiods = 1 / test_frequencies
				LSpowers = LSpowers * np.nanmax(np.abs(total_sinusoid)) #### making the power roughly equal to the amplitude.
		
				if show_plots == 'y':
					#### show it again, after removal
					fig, (ax1, ax2) = plt.subplots(2, figsize=(6,8))
					ax1.set_title('Post-removal, FAP = '+str(round(LS_FAP,2)))
					#ax1.plot(baseline_epochs, total_removal_test_sinusoid, c='k', linewidth=2, alpha=0.5)
					ax1.scatter(baseline_epochs[random_idxs], total_removal_test_sinusoid[random_idxs], facecolor='LightCoral', edgecolor='k', s=20, zorder=1)
					ax1.errorbar(baseline_epochs[random_idxs], total_removal_test_sinusoid[random_idxs], yerr=sinusoid_sigma, ecolor='k', fmt='none', zorder=0)				
					ax2.plot(LSperiods, LSpowers, c='r', linewidth=2, alpha=0.7, label='new')
					ax2.plot(LSperiods, last_LSpowers, c='k', linewidth=1, linestyle='--', alpha=0.5, label='old')
					ax2.set_xscale('log')
					#ax2.plot(np.linspace(baseline_epochs_frequency_sep,baseline_epochs_frequency_sep,100), np.linspace(0,1.05*np.nanmax(LSpowers),100), c='g', linestyle='--', label='cadence frequency')
					ax2.plot(LSperiods[local_idxs], LSpowers[local_idxs], c='r', linestyle='--')
					#for n,ta,tf,tph,tp in zip(nsignal_range,trial_amps, trial_freqs, trial_phases, trial_periods):
					#	ax2.plot(np.linspace(tp,tp,100), np.linspace(0,1.05*np.nanmax(LSpowers),100), color=colors[n], alpha=0.7)

					plt.legend(loc=1)
					plt.show()		
					print(' ')
					print(' ')	

			last_FAP = LS_FAP 
			nsignals.append(rand_nsignals)
			niterations.append(iteration)

			print("# of signals: ", rand_nsignals)
			print('# of iterations: ', iteration)	
			print(" ")
			###### END WHILE LOOP
			trial_dict[trial]['FAPs'] = np.array(trial_FAPs)	
			print("added trial_dict[trial]['FAPs']")
			#time.sleep(10)


			if write_to_file == 'y':
				with open('/data/tethys/Documents/Projects/NMoon_TTVs/iterative_periodogram_results.pkl', 'wb') as handle:
					pickle.dump(trial_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

		except:
			traceback.print_exc()
			continue









	nsignals, niterations = np.array(nsignals), np.array(niterations)

	FAP_test_range = np.logspace(-8,0,500)
	n1_frac_correct, n2_frac_correct, n3_frac_correct, n4_frac_correct = [], [], [], []

	for FAP_threshold in FAP_test_range:
		print("testing FAP threshold: ", FAP_threshold)
		n1correct, n2correct, n3correct, n4correct = 0, 0, 0, 0
		n1total, n2total, n3total, n4total = 0, 0, 0, 0
		### go through the dictionary, and look at the FAP for iteration = nmoons.
		for tdk in trial_dict.keys():
			try:
				trial_nmoons = trial_dict[tdk]['nsignals']
				one_signal_left_FAP = trial_dict[tdk]['FAPs'][trial_nmoons-1]
				completely_eliminated_signal_FAP = trial_dict[tdk]['FAPs'][trial_nmoons] ### index 0 is original FAP, index 1 is after first removal, index 2 after second removal, etc.
				
				if (completely_eliminated_signal_FAP >= FAP_threshold) and (one_signal_left_FAP < FAP_threshold):
					#### you STOPPED IN PRECISELY THE RIGHT PLACE.
					last_it_is_nmoons = True
				else:
					#### you got it WRONG. EITHER THE NEXT ONE ALSO BELOW THE THRESHOLD, OR THE PREVIOUS ONE WAS ALSO ABOVE THE THRESHOLD.
					last_it_is_nmoons = False

				if trial_nmoons == 1:
					n1total += 1
					if last_it_is_nmoons == True:
						n1correct += 1
				elif trial_nmoons == 2:
					n2total += 1
					if last_it_is_nmoons == True:
						n2correct += 1
				elif trial_nmoons == 3:
					n3total += 1
					if last_it_is_nmoons == True:
						n3correct += 1
				elif trial_nmoons == 4:
					n4total += 1
					if last_it_is_nmoons == True:
						n4correct += 1
			except:
				continue
		try:
			n1_frac_correct.append(n1correct / n1total)
		except:
			n1_frac_correct.append(np.nan)
		try:
			n2_frac_correct.append(n2correct / n2total)
		except:
			n2_frac_correct.append(np.nan)
		try:
			n3_frac_correct.append(n3correct / n3total)
		except:
			n3_frac_correct.append(np.nan)
		try:
			n4_frac_correct.append(n4correct / n4total)
		except:
			n4_frac_correct.append(n4correct / n4total)

	fig, ax = plt.subplots(1, figsize=(6,8))
	ax.plot(FAP_test_range, n1_frac_correct, color=colors[0], linewidth=2, label='N=1')
	ax.plot(FAP_test_range, n2_frac_correct, color=colors[1], linewidth=2, label='N=2')
	ax.plot(FAP_test_range, n3_frac_correct, color=colors[2], linewidth=2, label='N=3')
	ax.plot(FAP_test_range, n4_frac_correct, color=colors[3], linewidth=2, label='N=4')
	plt.xlabel('FAP threshold')
	plt.ylabel('fraction correct')
	plt.xscale('log')
	plt.legend(loc=0)
	plt.subplots_adjust(left=0.125, bottom=0.09, right=0.9, top=0.95, wspace=0.2, hspace=0.2)
	plt.show()




	for tdk in trial_dict.keys():
		#### look at the FAP curves, shift them over by the elimination iteration, such that 
		#### they're all stacked there. 
		#### there were 10 removals, which means there are 11 FAPs in each array.
		#### for N=3, the "elimination iteration" is the fourth one (0 removed, 1 removed, 2 removed, 3 removed).
		#### so we want plot this as plt.plot(np.arange(0,niterations,1)-trial_dict[tdk]['nsignals'], trial_dict[tdk]['FAPs'])
		try:
			plt.plot(np.arange(0,num_iterations+1,1)-trial_dict[tdk]['nsignals'], trial_dict[tdk]['FAPs'], c='k', alpha=0.2)
		except:
			continue
	plt.plot(np.linspace(0,0,100),np.logspace(-10,0,100), c='r', linestyle='--')
	plt.ylim(1e-10, 1e0)
	plt.xlabel('elimination iteration')
	plt.ylabel('FAP')
	plt.yscale('log')
	plt.show()




	nmoons = []
	last_FAPs = []
	for tdk in trial_dict.keys():
		nmoons.append(trial_dict[tdk]['nsignals'])
		last_FAPs.append(trial_dict[tdk]['FAPs'][-1])
	nmoons, last_FAPs = np.array(nmoons), np.array(last_FAPs)

	fig, ax = plt.subplots(4, sharex=True)
	for nm in np.arange(1,5,1):
		nmoon_idxs = np.where(nmoons == nm)[0]
		ax[nm-1].hist(last_FAPs[nmoon_idxs], bins=20, facecolor='DodgerBlue', edgecolor='k', alpha=0.7)
		ax[nm-1].set_ylabel('N = '+str(nm))

		print('N = '+str(nm))
		print("pct above FAP=0.9: ", )
	ax[nm-1].set_xlabel('False Alarm Probability')
	plt.show()




except:
	traceback.print_exc()
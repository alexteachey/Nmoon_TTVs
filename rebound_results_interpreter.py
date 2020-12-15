from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas
import os 
import time
import traceback
import pickle
from scipy.optimize import curve_fit
from astropy.timeseries import LombScargle 
from scipy.interpolate import interp1d 
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from astropy.constants import R_sun, G, M_sun
from keras.models import Sequential
from keras.layers import Dense 
from keras.utils import to_categorical
import socket


#### THIS CODE WILL ANALYZE THE OUTPUTS OF YOUR REBOUND SIMULATIONS FOR THE MULTI-MOON TTV ANALYSIS.


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



### you should really just save every simulation as a pickle! instead of initial conditions and positions. OK, whatev.

nsims = len(os.listdir(LSdir))


show_sys_plots = input("Do you want to show individual system plots? y/n: ")
run_planet_period_experiment = input('Do you want to run the planet period experiment? y/n: ')
run_LS_on_xpos = input('Do you want to run LombScargle on the xpositions? y/n: ')
keras_or_skl = input("Do you want to use 'k'eras or 's'cikit-learn? ")
if keras_or_skl == 's':
	mpl_or_rf = input("Do you want to use a 'm'ulti-layer perceptron, or a 'r'andom forest classifier? ")
#normalize_data = input("Do you want to normalize your data? y/n: ")
normalize_data = 'y'


def normalize(array):
	num = array - np.nanmin(array)
	denom = np.nanmax(array) - np.nanmin(array)
	return num / denom 



def build_MLP_inputs(*args, arrays='features'):
	### we're going to use this function to build a 2-D input_array
	### in a vertical stack, the shape = (nrows, ncolumns)
	#### for the MLP classifier, it has to be shape = n_samples, n_features
	##### that is, each ROW is a training example (sample), and each COLUMN is an input feature.
	###### what we're going to be LOADING IN, THOUGH, ARE ARRAYS OF FEATURES.
	###### THE LENGTH WILL BE EQUAL TO THE NUMBER OF SAMPLES (EXAMPLES)

	if arrays == 'features':
		#### means each array input is something like an array of Mplans, Pplans, etc.
		outstack = np.zeros(shape=(len(args[0]), len(args)))
		for narg, arg in enumerate(args):
			outstack.T[narg] = arg ### transposing so you can index by the column number

	elif arrays == 'examples':
		#### means each array is a series of features for a single input example. would be weird to do it this way, but I guess you could.
		outstack = np.zeros(shape=(len(args), len(args[0])))
		for narg, arg in enumerate(args):
			outstack[narg] = arg 

	return outstack




def MLP_classifier(input_array, target_classifications, hidden_layers=5, neurons_per_layer=100):
	#### input_array should be 2-Dimensional (shape=n_samples, n_features)
	assert input_array.shape[0] > input_array.shape[1] 
	#### you're gonna want more examples than features!

	assert len(target_classifications) == input_array.shape[0] ### every input sample should have a corresponding classification output!

	hidden_layer_neuron_list = []
	for i in np.arange(0,hidden_layers,1):
		hidden_layer_neuron_list.append(neurons_per_layer)
	hidden_layer_tuple = tuple(hidden_layer_neuron_list)

	clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=hidden_layer_tuple, verbose=True, early_stopping=True)

	clf.fit(input_array, target_classifications)

	return clf #### outputs the classifier that's ready to take inputs as, inputs.


def RF_classifier(input_array, target_classifications, n_estimators=100, max_depth=10, max_features=5):
	assert input_array.shape[0] > input_array.shape[1]
	assert len(target_classifications) == input_array.shape[0]

	clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features)
	clf.fit(input_array, target_classifications)

	return clf 






try:
	print('Do you want to load the MLP dictionary? ')
	print('Loading is faster, not loading will overwrite the old dictionary once everything is read in.')
	load_mlp = input('y / n: ')
	if load_mlp == 'y':
		mlp_dict = pickle.load(open(projectdir+'/MLP_dictionary.pkl', "rb"))
	else:
		pass 
	
	planet_masses = mlp_dict['Mplans']
	planet_periods = mlp_dict['Pplans']
	TTV_periods = mlp_dict['PTTV']
	TTV_rms = mlp_dict['TTV_rms']
	TTV_snrs = mlp_dict['TTV_snrs']

	#### (POSSIBLE) OUTPUTS
	nmoons = mlp_dict['nmoons']
	moon_masses = mlp_dict['Mmoons']
	moon_periods = mlp_dict['Pmoons']
	moon_fRHills = mlp_dict['fRHills']
	moon_ordernumbers = mlp_dict['Morder_nums']

	print('SUCCESSFULLY LOADED THE MLP DICTIONARY.')




except:
	print("UNABLE TO LOAD THE MLP DICTIONARY. WILL LOAD IN THE HARD (SLOW) WAY.")
	time.sleep(5)

	#### BUILD THE LIST OF INPUTS AND OUTPUTS FOR THE MLP!
	##### inputs
	planet_masses = [] #### kg
	planet_periods = [] ### days
	TTV_periods = [] ### epochs
	TTV_rms = []
	TTV_snrs = []

	#### target output
	nmoons = []
	moon_masses = np.array([])
	moon_periods = np.array([])
	moon_fRHills = np.array([])
	moon_ordernumbers = np.array([])


	for nsimnum, simnum in enumerate(np.arange(2,nsims,1)):
		print('simnum = ', simnum)
		try:
			fileprefix = 'TTVsim'+str(simnum)
			ttv_filename = fileprefix+'_TTVs.csv'
			xpos_filename = fileprefix+'_xpos.npy'
			ypos_filename = fileprefix+'_ypos.npy'
			model_dict_filename = fileprefix+'_system_dictionary.pkl'
			periodogram_filename = fileprefix+'_periodogram.npy' 

			### now load them!
			xpos, ypos = np.load(positionsdir+'/'+xpos_filename), np.load(positionsdir+'/'+ypos_filename)
			LSperiods, LSpowers = np.load(LSdir+'/'+periodogram_filename)
			ttvfile = pandas.read_csv(ttvfiledir+'/'+ttv_filename)
			model_dict = pickle.load(open(modeldictdir+'/'+model_dict_filename, "rb"))


			#raise Exception('play with the model_dict!')
			this_planet_period_seconds = model_dict['Planet']['P']
			this_planet_period_days = this_planet_period_seconds / (24 * 60 * 60)
			this_planet_tdur_approx_seconds = (this_planet_period_seconds / np.pi) * np.arcsin(R_sun.value / ((G.value * M_sun.value * this_planet_period_seconds**2) / (4*np.pi**2))**(1/3))
			this_planet_ttv_errs_seconds = 0.04*this_planet_tdur_approx_seconds


			#### ADDING TO OUR INPUTS
			print('appending to planet_masses and planet_periods...')
			planet_masses.append(model_dict['Planet']['m']) 
			planet_periods.append(this_planet_period_days)

			this_system_masses = []
			this_system_fRHills = []
			this_system_periods = []


			num_moons = 0
			for moon_name in np.array(['I','II', 'III', 'IV', 'V']):
				#### grab the values
				try:
					moon_mass = model_dict[moon_name]['m']
					moon_a = model_dict[moon_name]['a']
					moon_Psecs = model_dict[moon_name]['P']
					moon_fRHill = moon_a / model_dict['Planet']['RHill']

					this_system_masses.append(moon_mass)
					this_system_fRHills.append(moon_fRHill)
					this_system_periods.append(moon_Psecs)
					num_moons += 1
				except:
					pass


			### ADDING TO OUR CLASSIFICATION OUTPUTS.
			print('appending to nmoons...')
			nmoons.append(num_moons)


			fRHill_argsort = np.argsort(this_system_fRHills)
			this_system_sorted_masses = np.array(this_system_masses)[fRHill_argsort]
			this_system_sorted_fRHills = np.array(this_system_fRHills)[fRHill_argsort]
			this_system_sorted_periods = np.array(this_system_periods)[fRHill_argsort]
			this_system_moon_ordernumbers = np.arange(1,len(this_system_masses)+1,1)

			moon_masses = np.concatenate((moon_masses, this_system_sorted_masses))
			moon_periods = np.concatenate((moon_periods, this_system_sorted_periods))
			moon_fRHills = np.concatenate((moon_fRHills, this_system_sorted_fRHills))
			moon_ordernumbers = np.concatenate((moon_ordernumbers, this_system_moon_ordernumbers))



			#raise Exception('all you want to do right now.')



			#### PERIODOGRAM STUFF -- generated in rebound_playground.py
			# normalize the periodogram so the peak period is at 1.
			LSpowers = LSpowers / np.nanmax(LSpowers) 

			if simnum == 2: ### first one, for some reason -- anyway initialize this
				LSpowerstack = LSpowers

			LSpowerstack = np.vstack((LSpowerstack, LSpowers))

			### compute a running mean and median
			running_mean = np.nanmean(LSpowerstack, axis=0)
			running_median = np.nanmedian(LSpowerstack, axis=0)


			#### NOW get into the TTV file
			epochs = np.array(ttvfile['epoch'])
			ttvobs = np.array(ttvfile['TTVobs'])
			ttvrms = np.sqrt(np.nanmean(ttvobs**2))
			ttverrs = np.linspace(ttvrms, ttvrms, len(ttvobs))
			transit_times = np.array(ttvfile['tobs'])
			### fit the line, infer the period
			inferred_period = np.polyfit(epochs, transit_times, deg=1)[0]


			### pull out the best period!
			best_TTVperiod = LSperiods[np.nanargmax(LSpowers)]
			best_TTVfreq = 1/best_TTVperiod
			best_TTVangfreq = 2*np.pi*best_TTVfreq


			#### ADDING TO OUR INPUTS
			print('appending to TTV_periods...')
			TTV_periods.append(best_TTVperiod) ### IS IT IN EPOCHS?!





			def sinefit(times, amp, phase):
				### doing this within the loop because you want to fix angfreq
				return amp * np.sin(best_TTVangfreq*times + phase)



			#### FIT the sinusoid with curve fit
			try:
				popt, pcov = curve_fit(sinefit, epochs, ttvobs, bounds=([0, -2*np.pi], [5*ttvrms, 2*np.pi]))
				ttv_fit_amplitude, ttv_fit_phase = popt

				chi2_flat = np.nansum(ttvobs**2) / ttvrms 
				chi2_ttv = np.nansum( (epochs - sinefit(epochs, *popt) )**2 ) / ttvrms
				BIC_flat = chi2_flat
				BIC_ttv = 2*np.log(len(epochs)) + chi2_ttv

				if show_sys_plots == 'y':
					plt.scatter(epochs, ttvobs, facecolor='LightCoral', edgecolor='k', zorder=1)
					plt.errorbar(epochs, ttvobs, yerr=ttvrms, fmt='none', zorder=0, ecolor='k')
					epochs_interp = np.linspace(np.nanmin(epochs), np.nanmax(epochs), 1000)
					sinefit_interp = sinefit(epochs_interp, *popt)
					plt.plot(epochs_interp, sinefit_interp, color='r', linestyle='--', linewidth=2, zorder=2, label='BIC = '+str(np.round(BIC_ttv, 2)))
					plt.plot(epochs_interp, np.linspace(0,0,len(epochs_interp)), color='k', linestyle='--', linewidth=2, zorder=2, label='BIC = '+str(np.round(BIC_flat, 2)))
					plt.title(r'$\Delta$ BIC = '+str(np.round(BIC_ttv - BIC_flat,2)))
					plt.show()
			except:
				pass


			#### ADDING TO OUR INPUTS.
			print('appending to TTV_rms and TTV_snrs...')
			TTV_rms.append(ttvrms)
			TTV_snrs.append(ttv_fit_amplitude / this_planet_ttv_errs_seconds)



			#### your xpositions are the real underlying oscillation signal! plot those in units of epochs!
			run_period_years = 10
			run_period_days = run_period_years * 365.25 ### just as a test run 
			run_period_hours = run_period_days * 24
			run_period_minutes = run_period_hours * 60
			run_period_seconds = run_period_minutes * 60
			Noutputs = 10000 ### number of evaluations -- not necessarily the number of time steps from sim.dt!
			sim_times = np.linspace(0,run_period_seconds, Noutputs)
			sim_times_in_epochs = sim_times / inferred_period

			if show_sys_plots == 'y':
				plt.plot(sim_times_in_epochs, xpos[0]/np.nanmax(xpos[0]), color='DodgerBlue')
				plt.plot(epochs_interp, sinefit_interp/np.nanmax(sinefit_interp), color='red', linestyle='--')
				plt.xlabel('Epochs')
				plt.ylabel('physical displacement')
				#plt.set_ylabel(r'normalized $x$-displacement')
				plt.show()



			
			#### LET'S SEE WHAT THE INFERRED OSCILLATION LOOKS LIKE COMPARED TO THE ACTUAL OSCILLATION
			
			"""
			fig, (ax1, ax2) = plt.subplots(2, sharex=True)

			ax1.plot(sim_times_in_epochs, xpos/np.nanmax(xpos), color='DodgerBlue')
			ax1.set_ylabel(r'normalized $x$-displacement')
			
			ax2.scatter(epochs, ttvobs, facecolor='LightCoral', edgecolor='k', zorder=1)
			ax2.errorbar(epochs, ttvobs, yerr=ttvrms, fmt='none', zorder=0, ecolor='k')
			epochs_interp = np.linspace(np.nanmin(epochs), np.nanmax(epochs), 1000)
			ax2.plot(epochs_interp, sinefit(epochs_interp, *popt), color='r', linestyle='--', linewidth=2)
			plt.show()
			"""
			
			

			#### run a periodogram on the xpos oscillation and compare to the TTV periodogram!
		
			if run_LS_on_xpos == 'y':
				xpos_period_min = 1 ### second
				xpos_period_max = 30 * 24 * 60 * 60 ### 30 days in seconds
				xpos_LS_frequencies = np.logspace(np.log10(1/xpos_period_max), np.log10(1/xpos_period_min), 1000)
				xpos_LS_periods = 1 / xpos_LS_frequencies
				xpos_LS_powers = LombScargle(sim_times, xpos[0]).power(xpos_LS_frequencies)

				best_xpos_LS_freq = xpos_LS_frequencies[np.argmax(xpos_LS_powers)]
				best_xpos_LS_period = 1/best_xpos_LS_freq

				xpos_LS = LombScargle(sim_times, xpos[0])
				best_fit = xpos_LS.model(sim_times, best_xpos_LS_freq)

				if show_sys_plots == 'y':
					for tsmon, tsp in zip(this_system_moon_ordernumbers, this_system_sorted_periods):
						plt.plot(np.linspace(tsp/(60*60*24), tsp/(60*60*24), 100), np.linspace(0,1,100), linestyle='--', alpha=0.5, label=str(tsmon))

					plt.plot(xpos_LS_periods/(60 * 60 * 24), xpos_LS_powers/np.nanmax(xpos_LS_powers), c='k')
					plt.xlabel('days')
					plt.xscale('log')
					plt.legend()
					plt.show()
			


			### CALL THIS THE "PLANET PERIOD EXPERIMENT" -- hence, ppe 
			#### THE IDEA IS / WAS -- CAN WE SEE SOME PATTERN IN THE PERIODOGRAM BEHAVIOR AS WE GO FROM HIGH SAMPLING RATE
			#### (SHORT PERIOD PLANET) - to LOW SAMPLING RATE (LONG PERIOD PLANET) -- doesn't seem to have been addressed in
			#### DAVID'S PAPER -- seems to have a big effect though... QUESTION IS, DO WE NEED TO NORMALIZE THESE BY PLANET PERIOD.

			#### THIS SHIT IS REALLY WEIRD -- HOW DOES IT SQUARE WITH THE SINGLE MOON CASE???

			#### WITH A SINGLE MOON, YOU OUGHT TO BE GET THE SAME SINUSOID NO MATTER WHEN YOU SAMPLE IT... CAN YOU TEST THAT???


			### single moon_case
			#### THE KEY IS NORMALIZING THE INFERRED PERIOD BY THE ORBITAL PERIOD OF THE PLANET!!!!
			##### THAT'S WHERE THE PATTERN EMERGES -- YOU'LL ***NEVER*** FIT THE CORRECT PERIOD, BECAUSE
			###### THE MOON PERIOD WILL ***ALWAYS*** BE SHORTER THAN THE ORBITAL PERIOD OF THE PLANET, AND
			####### YOU SIMPLY CANNOT PROBE THESE FREQUENCIES -- IT'S NONSENSE!!!!!!
			######## EXAMINING THE SINGLE-SINUSOID CASE WAS ILLUMINATING.

			"""
			ss_periods = np.arange(10,1500,5.234235) ### single sinusoid periods
			ss_best_LS_periods = []

			for ssp in ss_periods:
				ss_times = np.arange(0,1000,ssp)

				ss_moon_period = 7.23423
				ss_linfreq = 1/ss_moon_period
				ss_angfreq = 2*np.pi*ss_linfreq #### orbital period is 5 days!
				single_moon_sinusoid = np.sin(ss_angfreq*ss_times) 
				ss_min_period = ssp
				ss_max_period = 100*ss_min_period
				ss_min_angfreq = (2 * np.pi) / ss_max_period
				ss_max_angfreq = (2 * np.pi) / ss_min_period 

				ss_sample_frequencies = np.logspace(np.log10(ss_min_angfreq), np.log10(ss_max_angfreq), 10000)

				#### run a periodogram on these positions!
				ss_LS_powers = LombScargle(ss_times, single_moon_sinusoid).power(ss_sample_frequencies)
				ss_best_LS_freq = ss_sample_frequencies[np.argmax(ss_LS_powers)]
				ss_best_LS_angfreq = 2*np.pi * ss_best_LS_freq
				ss_best_LS_period = 1 / ss_best_LS_freq 
				ss_best_LS_periods.append(ss_best_LS_period)

				#### try to fit a a sinusoid to it
				def ss_sinefit(times, amp, phase):
					### doing this within the loop because you want to fix angfreq
					return amp * np.sin(ss_best_LS_angfreq*times + phase)

				np.vectorize(ss_sinefit)

				#### LOMB-SCARGLE ANGULAR FREQUENCY IS HARD CODED IN HERE! ONLY PHASE AND AMPLITUDE ARE BEING FIT!
				try:
					popt, pcov = curve_fit(ss_sinefit, ss_times, single_moon_sinusoid, bounds=([0, -2*np.pi], [10*np.nanmax(single_moon_sinusoid), 2*np.pi]))

					ss_interptimes = np.linspace(np.nanmin(ss_times), np.nanmax(ss_times), 1000)
					plt.scatter(ss_times, single_moon_sinusoid, facecolor='DodgerBlue', edgecolor='k', s=10, alpha=0.5)
					plt.plot(ss_interptimes, np.sin(ss_angfreq*ss_interptimes), c='k', linestyle=':', label='actual', alpha=0.5)
					plt.plot(ss_interptimes, ss_sinefit(ss_interptimes, *popt), c='red', linestyle='--', label='inferred', alpha=0.5)
					#plt.title(r'$P_P$ = '+str(np.round(ssp, 2))+', fit $P_S$= '+str(np.round(ss_best_LS_period,2))+r', actual $P_S$ = '+str(7.23))
					plt.title(r'$P_P$ = '+str(np.round(ssp,2))+r', fit $P_S$ = '+str(np.round((ss_best_LS_period / ssp),2)+' epochs'))
					plt.show()
				except:
					pass 

			"""



			if run_planet_period_experiment == 'y':
				#### probing from 1/10th the orbital period of this simulated planet
				ppe_periods_days = np.linspace(0.1*this_planet_period_days, this_planet_period_days, 20)
				ppe_best_LS_periods_days = []
				ppe_best_LS_periods_epochs = []
				num_ppe_periods = len(ppe_periods_days)


				#### GONNA TRY SOMETHING NEW (NOVEMBER 5th) --
				##### we want to see how the INFERRED TTV PERIOD (in EPOCHS) IS CONNECTED TO THE ORBITAL PERIOD OF THE PLANET, FOR THE SAME SET OF MOONS!!
				###### AND WE WANT TO FURTHER CONNECT THAT TO THE ORBITAL PERIODS OF THE MOONS!


				for npp, pp in enumerate(ppe_periods_days):  ### days

					### higher sampling will be cleaner periodicity -- maybe!

					#### we're going to sample xpos only at these these time steps!
					#### see what happens to the periodogram! AND, what our inferrences become
					#### NOTE THAT SIM TIMES IS IN SECONDS!

					planet_period_in_seconds = pp * (24 * 60 * 60)
					sample_times = np.arange(np.nanmin(sim_times), np.nanmax(sim_times), planet_period_in_seconds) ### THESE ARE TRANSIT TIMES! SECONDS!
					sample_times_days = sample_times / (24 * 60 * 60)


					xpos_interpolator = interp1d(sim_times, xpos[0]) ### interpolates the x-positions
					ypos_interpolator = interp1d(sim_times, ypos[0])
					interpolated_xpositions = xpos_interpolator(sample_times)
					interpolated_ypositions = ypos_interpolator(sample_times)

					ppe_min_probed_period_days = 2*pp
					ppe_max_probed_period_days = 100*pp 

					ppe_min_probed_linfreq = 1/ppe_max_probed_period_days
					ppe_max_probed_linfreq = 1/ppe_min_probed_period_days

					ppe_min_probed_angfreq = 2*np.pi*ppe_min_probed_linfreq
					ppe_max_probed_angfreq = 2*np.pi*ppe_max_probed_linfreq

					ppe_linfreq_range = np.linspace(ppe_min_probed_linfreq, ppe_max_probed_linfreq, 10000)

					#### run a periodogram on these positions!
					ppe_LS_powers = LombScargle(sample_times_days, interpolated_xpositions).power(ppe_linfreq_range)
					ppe_probed_periods = 1/ppe_linfreq_range
					ppe_best_LS_linfreq = ppe_linfreq_range[np.argmax(ppe_LS_powers)]
					ppe_best_LS_angfreq = 2*np.pi*ppe_best_LS_linfreq
					ppe_best_LS_period_days = 1 / ppe_best_LS_linfreq  #### seconds!
					ppe_best_LS_periods_days.append(ppe_best_LS_period_days)
					ppe_best_LS_periods_epochs.append(ppe_best_LS_period_days / pp)



					#### plot the x and y positions at the sample times, make sure this is working right!

					#fig, (ax1, ax2) = plt.subplots(2)
					#ax1.scatter(interpolated_xpositions, interpolated_ypositions, c='DodgerBlue', edgecolor='k', s=20)
					#ax1.scatter(0, 0, marker='X', s=100, color='red', alpha=0.5)


					def ppe_sinefit(times, amp, phase):
						### doing this within the loop because you want to fix angfreq
						return amp * np.sin(ppe_best_LS_angfreq*times + phase)


					try:
						#if (ppe_best_LS_period_days / pp) > 10:

						if show_sys_plots == 'y':
							fig, (ax1, ax2) = plt.subplots(2)

							ax1.scatter(sample_times_days, interpolated_xpositions, c='DodgerBlue', edgecolor='k', s=20)
							ax1.set_xlabel('days')
							ax1.set_ylabel('x-diplacement')

							popt, pcov = curve_fit(ppe_sinefit, sample_times_days, interpolated_xpositions, bounds=([0, -2*np.pi], [10*np.nanmax(interpolated_xpositions), 2*np.pi]))

							ax1.plot(sample_times_days, ppe_sinefit(sample_times_days, *popt), c='r', linestyle='--')
							

							ax2.plot(ppe_probed_periods, ppe_LS_powers, c='k', alpha=0.5) 
							ax2.set_xscale('log')

							this_system_sorted_periods_days = this_system_sorted_periods / (24 * 60 * 60)
							for tsmon, tsp in zip(this_system_moon_ordernumbers, this_system_sorted_periods_days):
								#ax2.plot(np.linspace(tsp/pp, tsp/pp, 100), np.linspace(0,1,100), linestyle='--', alpha=0.5, label=str(tsmon))
								ax2.plot(np.linspace(tsp, tsp, 100), np.linspace(0,1,100), linestyle='--', alpha=0.5, label=str(tsmon))

							#plt.xlabel('x-displacement from CoM')
							#plt.ylabel('y-displacement from CoM')
							#plt.title('planet period = '+str(pp)+' days')
							ax1.set_title(r'$P_P$ = '+str(np.round(pp,2))+r' days, $P_{TTV}$ = '+str(np.round((ppe_best_LS_period_days / pp),2))+' epochs')
							plt.show()


					except:
						pass






		
				"""
				#### PLOTTING THE PERIODOGRAMS -- just gets hairier and hairier!
				if show_sys_plots == 'y':
					for tsmon, tsp in zip(this_system_moon_ordernumbers, this_system_sorted_periods):
						plt.plot(np.linspace(tsp/(60*60*24), tsp/(60*60*24), 100), np.linspace(0,1,100), linestyle='--', alpha=0.5, label=str(tsmon))

					plt.title('Period (days) = '+str(pp)+', #obs = '+str(len(sample_times)))
					plt.plot(1/xpos_LS_frequencies, ppe_LS_powers/np.nanmax(ppe_LS_powers), c='k')
					#plt.xlabel('days')
					plt.xscale('log')
					plt.legend()
					plt.show()
				"""
				

			

			#plt.plot(xpos_LS_periods/(60*60*24), ppe_LS_powers/np.nanmax(ppe_LS_powers), alpha=0.5)


			#### PLOTTING THE PEAK POWER PERIODS FROM LOMB SCARGLE AT DIFFERENT SAMPLING CADENCES@
			
			"""
			ppe_best_LS_periods_days = np.array(ppe_best_LS_periods_days)
			this_system_sorted_periods_days = this_system_sorted_periods / (24 * 60 * 60)

			if show_sys_plots == 'y':

				plt.scatter(ppe_periods_days, (ppe_best_LS_periods_days/ppe_periods_days), color='LightCoral', edgecolor='k', alpha=0.5, s=20)

				#### PLOT THE MOONS
				#for tsmon, tsp in zip(this_system_moon_ordernumbers, this_system_sorted_periods_days):
				#	plt.plot(np.linspace(np.nanmin(ppe_periods_days), np.nanmax(ppe_periods_days), 100), np.linspace(tsp, tsp, 100), linestyle='--', alpha=0.5, label=str(tsmon))

				plt.xscale('log')
				#plt.yscale('log')
				plt.xlabel('Planet Period / sampling rate (days)')
				#plt.ylim(1e-5, 1e2)
				plt.ylabel('TTV Period (Epoch)')
				plt.title('periodogram sampling evolution')
				plt.show()
			"""






			### now plot the TTVs with the resulting curve_fit

			"""
			#### FIX THIS PHASE FOLDING THING!
			if show_sys_plots == 'y':
				fig, (ax1, ax2) = plt.subplots(2, sharey=True)

				ax1.scatter(epochs, ttvobs, facecolor='LightCoral', edgecolor='k', zorder=1)
				ax1.errorbar(epochs, ttvobs, yerr=ttvrms, fmt='none', zorder=0, ecolor='k')
				epochs_interp = np.linspace(np.nanmin(epochs), np.nanmax(epochs), 1000)
				ax1.plot(epochs_interp, sinefit(epochs_interp, *popt), color='r', linestyle='--', linewidth=2)
				
				#### phase-fold!
				epochs_phasefold = epochs % (2*best_TTVangfreq * epochs)
				epochs_interp_phasefold = np.linspace(0, 2*best_TTVangfreq, 1000)
				ax2.scatter(epochs_phasefold, ttvobs, facecolor='LightCoral', edgecolor='k', zorder=1)
				ax2.errorbar(epochs_phasefold, ttvobs, yerr=ttvrms, fmt='none', zorder=0, ecolor='k')
				ax2.plot(epochs_interp_phasefold, sinefit(epochs_interp_phasefold, *popt), color='r', linestyle='--', linewidth=2)

				plt.show()
			"""




			#### NOW WE WANT TO DO THE FOLLOWING: 
			### 1. make a mean and median periodogram for all sims -- CHECK
			### 2. make a histogram of inputs
			### 3. make a histogram of resulting moon TTVs (peak period)
			### 4. maybe make some other plots about the moon positions, or something.
			### 4. FIT SINUSOIDS TO EACH TTV.


			### MAKE A HISTOGRAM OF THE RESULTING PERIODS!


			try:
				if nsimnum == 0:
					master_ppe_period_days_stack = np.array(ppe_periods_days)
					master_ppe_pttv_epochs_stack = np.array(ppe_best_LS_periods_epochs)

				else:
					master_ppe_period_days_stack = np.vstack((master_ppe_period_days_stack, np.array(ppe_periods_days)))
					master_ppe_pttv_epochs_stack = np.vstack((master_ppe_pttv_epochs_stack, np.array(ppe_best_LS_periods_epochs)))
				

				print('COMPLETED THE LOOP (TRY).')
				print(' ')

			except:
				print('COMPLETED THE LOOP (EXCEPT).')
				print(' ')

				continue



		except:
			traceback.print_exc()
			time.sleep(5)
			try:
				#for i in np.arange(0,master_ppe_period_days_stack.shape[0],1):
				#	plt.plot(master_ppe_period_days_stack[i], master_ppe_pttv_epochs_stack[i], alpha=0.25)

				plt.scatter(master_ppe_period_days_stack, master_ppe_pttv_epochs_stack, s=20, alpha=0.5, facecolor='DodgerBlue', edgecolor='k')

				plt.xlabel(r'$P_P$ [days]')
				plt.ylabel(r'$P_{TTV}$ [epochs]')
				plt.xscale('log')
				plt.show()
			except:
				#traceback.print_exc()
				continue 



			#traceback.print_exc()



			#raise Exception('something went wrong.')





	#### MLP SHIT
	##### inputs
	planet_masses = np.array(planet_masses) #### kg -- 1D array
	planet_periods = np.array(planet_periods) ### days -- 1D array
	TTV_periods = np.array(TTV_periods) ### epochs -- 1D array
	TTV_rms = np.array(TTV_rms) ### 1D array
	TTV_snrs = np.array(TTV_snrs) #### 1D array
	#### target output
	nmoons = np.array(nmoons)


	#### GENERATE THE DICTIONARY
	##### INPUTS: 
	mlp_dict = {}
	mlp_dict['Mplans'] = planet_masses
	mlp_dict['Pplans'] = planet_periods
	mlp_dict['PTTV'] = TTV_periods
	mlp_dict['TTV_rms'] = TTV_rms
	mlp_dict['TTV_snrs'] = TTV_snrs 

	#### (POSSIBLE) OUTPUTS
	mlp_dict['nmoons'] = nmoons 
	mlp_dict['Mmoons'] = moon_masses
	mlp_dict['Pmoons'] = moon_periods 
	mlp_dict['fRHills'] = moon_fRHills
	mlp_dict['Morder_nums'] = moon_ordernumbers



	#### save the dictionary!
	pickle.dump(mlp_dict, open(projectdir+'/MLP_dictionary.pkl', 'wb'))







try:

	##### NOW GO WITH THE MLP!
	MLP_continue = input('READY TO RUN THE MLP? y/n: ')
	if MLP_continue != 'y':
		raise Exception('you opted not to continue.')


	#### first we need to balance the training set across all the system types (2,3,4,5 moons).
	##### we know that five moons are the most rare, but let's make it more general.

	n1moons = len(np.where(nmoons == 1)[0])
	n2moons = len(np.where(nmoons == 2)[0])
	n3moons = len(np.where(nmoons == 3)[0])
	n4moons = len(np.where(nmoons == 4)[0])
	n5moons = len(np.where(nmoons == 5)[0])

	#### don't use n1moons here until you've made lots of them in the sims!
	nmoons_least_represented = np.nanmin((n2moons, n3moons, n4moons, n5moons)) / 2 #### divide 2 so you have half in the validation sample!

	n1, n2, n3, n4, n5 = 0, 0, 0, 0, 0 
	training_idxs = []
	validation_idxs = []

	for nmoon_idx, nmoon in enumerate(nmoons):
		if (nmoon == 1):
			ntest = n1
			n1 += 1
		elif (nmoon == 2):
			ntest = n2
			n2 += 1
		elif nmoon == 3:
			ntest = n3
			n3 += 1
		elif nmoon == 4:
			ntest = n4
			n4 += 1
		elif nmoon == 5:
			ntest = n5
			n5 += 1

		if ntest < nmoons_least_represented:
			training_idxs.append(nmoon_idx)
		elif ntest >= nmoons_least_represented:
			validation_idxs.append(nmoon_idx)

	training_idxs, validation_idxs = np.array(training_idxs), np.array(validation_idxs)

	print('# training samples = ', len(training_idxs))
	print('# validation samples = ', len(validation_idxs))




	if normalize_data == 'n':
		MLP_input_array = build_MLP_inputs(planet_masses, planet_periods, TTV_periods, TTV_rms, TTV_snrs)
	
	elif normalize_data == 'y':
		normed_planet_masses = normalize(planet_masses)
		normed_planet_periods = normalize(planet_periods)
		normed_TTV_periods = normalize(TTV_periods)
		normed_TTV_rms = normalize(TTV_rms)
		normed_TTV_snrs = normalize(TTV_snrs)
		MLP_input_array = build_MLP_inputs(normed_planet_masses, normed_planet_periods, normed_TTV_periods, normed_TTV_rms, normed_TTV_snrs)

	hidden_layer_options = np.arange(1,30,1)
	neurons_per_layer_options = np.arange(10,110,10)
	n_estimator_options = np.arange(10,110,10)
	max_depth_options = np.arange(1,21,1)
	max_features_options = np.arange(2,6,1)


	#### STARTING VALUES -- WILL BE UPDATED DURING THE LOOP!
	best_hlo = 0 ### hidden layer options
	best_nplo = 0 ### neurons_per_layer options
	best_neo = 0 ### best n_estimator options
	best_mdo = 0  #### best max_depth_options
	best_mfo = 0 #### best max_features_options

	best_accuracy = 0 
	best_categorical_accuracy = 0
	best_n1_accuracy = 0
	best_n2_accuracy = 0
	best_n3_accuracy = 0
	best_n4_accuracy = 0
	best_n5_accuracy = 0

	total_or_categorical = input("Do you want to optimize 't'otal or 'c'ategorical accuracy? ")

	if keras_or_skl == 'k':
		MLP_filename = 'keras_MLP_run.csv'

	elif keras_or_skl == 's':
		if mpl_or_rf == 'm':
			MLP_filename = 'sklearn_MLP_run.csv'
		elif mpl_or_rf == 'r':
			MLP_filename = 'sklearn_RF_run.csv'

	if os.path.exists(MLP_filename):
		#### open it and read the last hidden layer and neurons_per_layer -- first and second entries
		MLPfile = pandas.read_csv(projectdir+'/'+MLP_filename)
		if (keras_or_skl == 'k') or ((keras_or_skl == 's') and (mpl_or_rf == 'm')):
			last_hl = np.array(MLPfile['num_layers'])[-1]
			last_npl = np.array(MLPfile['neurons_per_layer'])[-1]
		elif (keras_or_skl == 's') and (mpl_or_rf == 'r'):
			last_neo = np.array(MLPfile['num_estimators'])[-1]
			last_md = np.array(MLPfile['max_depth'])[-1]
			last_mf = np.array(MLPfile['max_features'])[-1]


	else:
		last_hl = -1
		last_npl = -1 
		last_neo = -1
		last_md = -1
		last_mf = -1 

		MLPfile = open(projectdir+'/'+MLP_filename, mode='w')

		if (keras_or_skl == 'k') or ((keras_or_skl == 's') and (mpl_or_rf == 'm')):
			MLPfile.write('num_layers,neurons_per_layer,total_valacc,n1_actual,n1_preds,n1_precision,n1_recall,n2_actual,n2_preds,n2_precision,n2_recall,n3_actual,n3_preds,n3_precision,n3_recall,n4_actual,n4_preds,n4_precision,n4_recall,n5_actual,n5_preds,n5_precision,n5_recall\n')
		elif (keras_or_skl == 's') and (mpl_or_rf == 'r'):
			MLPfile.write('num_estimators,max_depth,max_features,total_valacc,n1_actual,n1_preds,n1_precision,n1_recall,n2_actual,n2_preds,n2_precision,n2_recall,n3_actual,n3_preds,n3_precision,n3_recall,n4_actual,n4_preds,n4_precision,n4_recall,n5_actual,n5_preds,n5_precision,n5_recall\n')


		MLPfile.close()



	if keras_or_skl == 'k':
		loop1 = hidden_layer_options
		loop2 = neurons_per_layer_options
		loop3 = np.array([1])

	elif (keras_or_skl == 's') and (mpl_or_rf == 'm'):
		loop1 = hidden_layer_options
		loop2 = neurons_per_layer_options
		loop3 = np.array([1])

	elif (keras_or_skl == 's') and (mpl_or_rf == 'r'):
		loop1 = n_estimator_options 
		loop2 = max_depth_options 
		loop3 = max_features_options




	"""
	START THE MASSIVE LOOP OF HYPERPARAMETERS!!!! 
	STARTS BELOW.
	"""

	#for hlo in hidden_layer_options:
	for l1 in loop1:
		if (keras_or_skl == 'k') or ((keras_or_skl == 's') and (mpl_or_rf == 'm')):
			hlo = l1 ### hidden layer option 
			if hlo < last_hl:
				continue

		elif (keras_or_skl == 's') and (mpl_or_rf == 'r'):
			neo = l1 ### n_estimator option 
			if neo < last_neo:
				continue 

		#for nplo in neurons_per_layer_options:
		for l2 in loop2:
			if (keras_or_skl == 'k') or ((keras_or_skl == 's') and (mpl_or_rf == 'm')):
				nplo = l2 ### neurons per layer option 
				if nplo < last_npl:
					continue

			elif (keras_or_skl == 's') and (mpl_or_rf == 'r'):
				mdo = l2 ### max_depth option
				if mdo < last_md:
					continue 


			for l3 in loop3:
				if (keras_or_skl == 'k') or ((keras_or_skl == 's') and (mpl_or_rf == 'm')):
					dummy_variable = l3 ### neurons per layer option 

				elif (keras_or_skl == 's') and (mpl_or_rf == 'r'):
					mfo = l3 ### max_features option
					if mfo < last_mf:
						continue 


				#### start a new row!
				MLPfile = open(projectdir+'/'+MLP_filename, mode='a')

				if (keras_or_skl == 'k') or ((keras_or_skl == 's') and (mpl_or_rf == 'm')):
					print('Number of hidden layers = ', hlo)
					print('Number of neurons per layer = ', nplo)

				elif (keras_or_skl == 's') and (mpl_or_rf == 'r'):
					print('Number of estimators = ', neo)
					print('Maximum depth = ', mdo)
					print("Maximum features = ", mfo)


				#time.sleep(1)

				if keras_or_skl == 's':
					### calling your own function, which fits the data under the hood.
					if mpl_or_rf == 'm':
						classifier = MLP_classifier(MLP_input_array[training_idxs], nmoons[training_idxs], hidden_layers=hlo, neurons_per_layer=nplo)
					
					elif mpl_or_rf == 'r':
						classifier = RF_classifier(MLP_input_array[training_idxs], nmoons[training_idxs])


				elif keras_or_skl == 'k':
					model = Sequential()
					for hlnum in np.arange(0,hlo,1):
						#### for every layer you're adding
						#model.add(Dense(nplo, input_layer=5, activation='relu'))
						model.add(Dense(nplo, activation='relu'))

					model.add(Dense(6, activation='sigmoid'))

					model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

					#### train this sucker
					model.fit(MLP_input_array[training_idxs], nmoons[training_idxs], epochs=100, batch_size=10)


				### VALIDATE IT!

				nhits = 0
				nmisses = 0
				ntotal = 0 

				### categorical accuracy -- maybe optimize this instead!!!!

				###### NOTE: PRECISION = TP / (TP + FP) --> nhits = TP, npreds == TP + FP, so precision = nhits / npreds
				####### RECALL = TP / TP + FN --> nhits == TP, ntotal = TP + FN so recall = nhits / ntotal.

				n1hits, n1preds, n1total, n1_accuracy = 0, 0, 0, 0 
				n2hits, n2preds, n2total, n2_accuracy = 0, 0, 0, 0
				n3hits, n3preds, n3total, n3_accuracy = 0, 0, 0, 0
				n4hits, n4preds, n4total, n4_accuracy = 0, 0, 0, 0
				n5hits, n5preds, n5total, n5_accuracy = 0, 0, 0, 0

				for nsample, sample in enumerate(MLP_input_array[validation_idxs]):
					#### make a prediction!
					actual_num_moons = nmoons[validation_idxs][nsample]
					
					if keras_or_skl == 's':
						
						if mpl_or_rf == 'm':
							try:
								classification = classifier.predict(sample)[0]
							except:
								sample = sample.reshape(1,-1)
								classification = classifier.predict(sample)[0]

						elif mpl_or_rf == 'r':
							try:
								classification = classifier.predict(sample)[0]
							except:
								sample = sample.reshape(1,-1)
								classification = classifier.predict(sample)[0]


					elif keras_or_skl == 'k':
						try:
							classification = model.predict_classes(sample)[0]
						except:
							sample = sample.reshape(1,-1)
							classification = model.predict_classes(sample)[0]



					#print('classification: ', classification)
					if classification == 1:
						n1preds += 1
					elif classification == 2:
						n2preds += 1
					elif classification == 3:
						n3preds += 1
					elif classification == 4:
						n4preds += 1
					elif classification == 5:
						n5preds += 1

					#print('actual: ', actual_num_moons)
					if classification == actual_num_moons: ### TP
						nhits += 1
						if actual_num_moons == 1:
							n1hits += 1
						elif actual_num_moons == 2:
							n2hits += 1
						elif actual_num_moons == 3:
							n3hits += 1
						elif actual_num_moons == 4:
							n4hits += 1
						elif actual_num_moons == 5:
							n5hits += 1

						#print('HIT!')
					else: #### 
						nmisses += 1
						#print("MISS!")

					if actual_num_moons == 1:
						n1total += 1
						n1_accuracy = n1hits / n1total 
					elif actual_num_moons == 2:
						n2total += 1
						n2_accuracy = n2hits / n2total
					elif actual_num_moons == 3:
						n3total += 1
						n3_accuracy = n3hits / n3total 
					elif actual_num_moons == 4:
						n4total += 1
						n4_accuracy = n4hits / n4total
					elif actual_num_moons == 5:
						n5total += 1 
						n5_accuracy = n5hits / n5total 

					ntotal += 1

					running_accuracy = nhits / ntotal
					running_categorical_accuracy = np.nanmean((n1_accuracy, n2_accuracy, n3_accuracy, n4_accuracy, n5_accuracy))

					##### END OF EXAMPLE LOOP

				try:
					n1precision = n1hits / n1preds
					n1recall = n1hits / n1total
				except:
					n1precision, n1recall = 0, 0

				try:
					n2precision = n2hits / n2preds
					n2recall = n2hits / n2total
				except:
					n2precision, n2recall = 0, 0

				try:
					n3precision = n3hits / n3preds
					n3recall = n3hits / n3total
				except:
					n3precision, n3recall = 0, 0

				try:
					n4precision = n4hits / n4preds
					n4recall = n4hits / n4total
				except:
					n4precision, n4recall = 0, 0

				try:
					n5precision = n5hits / n5preds
					n5recall = n5hits / n5total
				except:
					n5precision, n5recall = 0, 0





				accuracy = running_accuracy
				categorical_accuracy = running_categorical_accuracy 
				print('accuracy = ', accuracy*100)
				print('categorical accuracy = ', categorical_accuracy*100)
				print('n1 precision / recall = ', n1precision, n1recall)
				print('n2 precision / recall = ', n2precision, n2recall)
				print('n3 precision / recall = ', n3precision, n3recall)
				print('n4 precision / recall = ', n4precision, n4recall)
				print('n5 precision / recall = ', n5precision, n5recall)


				#### BEST VALUES UPDATES
				if accuracy > best_accuracy: ### improved 
					best_accuracy = accuracy 
					if total_or_categorical == 't':
						if (keras_or_skl == 'k') or ((keras_or_skl == 's') and (mpl_or_rf == 'm')):
							best_hlo = hlo 
							best_nplo = nplo 
						elif (keras_or_skl == 's') and (mpl_or_rf == 'r'):
							best_neo = neo 
							best_mdo = mdo 
							best_mfo = mfo 


				if categorical_accuracy > best_categorical_accuracy: ### improved 
					best_categorical_accuracy = categorical_accuracy 
					if total_or_categorical == 'c':
						if (keras_or_skl == 'k') or ((keras_or_skl == 's') and (mpl_or_rf == 'm')):
							best_hlo = hlo 
							best_nplo = nplo 
						elif (keras_or_skl == 's') and (mpl_or_rf == 'r'):
							best_neo = neo 
							best_mdo = mdo 
							best_mfo = mfo 

				if n1_accuracy > best_n1_accuracy:
					best_n1_accuracy = n1_accuracy
				if n2_accuracy > best_n2_accuracy:
					best_n2_accuracy = n2_accuracy
				if n3_accuracy > best_n3_accuracy:
					best_n3_accuracy = n3_accuracy
				if n4_accuracy > best_n4_accuracy:
					best_n4_accuracy = n4_accuracy
				if n5_accuracy > best_n5_accuracy:
					best_n5_accuracy = n5_accuracy 

	
				#MLPfile.write('num_layers,neurons_per_layer,total_valacc,n1_actual,n1_preds,n1_precision,n2_recall,n2_actual,n2_preds,n2_precision,n2_recall,n3_actual,n3_preds,n3_precision,n3_recall,n4_actual,n4_preds,n4_precision,n4_recall,n5_actual,n5_preds,n5_precision,n5_recall\n')	
				if (keras_or_skl == 'k') or ((keras_or_skl == 's') and (mpl_or_rf == 'm')):
					MLPfile.write(str(hlo)+','+str(nplo)+','+str(accuracy)+','+str(n1total)+','+str(n1preds)+','+str(n1precision)+','+str(n1recall)+','+str(n2total)+','+str(n2preds)+','+str(n2precision)+','+str(n2recall)+','+str(n3total)+','+str(n3preds)+','+str(n3precision)+','+str(n3recall)+','+str(n4total)+','+str(n4preds)+','+str(n4precision)+','+str(n4recall)+','+str(n5total)+','+str(n5preds)+','+str(n5precision)+','+str(n5recall)+'\n')

				elif (keras_or_skl == 's') and (mpl_or_rf == 'r'):
				#MLPfile.write('num_estimators,max_depth,max_features,total_valacc,n1_actual,n1_preds,n1_precision,n1_recall,n2_actual,n2_preds,n2_precision,n2_recall,n3_actual,n3_preds,n3_precision,n3_recall,n4_actual,n4_preds,n4_precision,n4_recall,n5_actual,n5_preds,n5_precision,n5_recall\n')
					MLPfile.write(str(neo)+','+str(mdo)+','+str(mfo)+','+str(accuracy)+','+str(n1total)+','+str(n1preds)+','+str(n1precision)+','+str(n1recall)+','+str(n2total)+','+str(n2preds)+','+str(n2precision)+','+str(n2recall)+','+str(n3total)+','+str(n3preds)+','+str(n3precision)+','+str(n3recall)+','+str(n4total)+','+str(n4preds)+','+str(n4precision)+','+str(n4recall)+','+str(n5total)+','+str(n5preds)+','+str(n5precision)+','+str(n5recall)+'\n')

				MLPfile.close()



				#time.sleep(5)




	print(" ")
	print(" X X X X X ")
	print(" SUMMARY ")
	print(" X X X X X ")
	print(' ')
	print('best accuracy so far: ', best_accuracy)
	print("best categorical accuracy so far: ", best_categorical_accuracy)
	print('best # hidden layers = ', best_hlo)
	print('best # neurons per layer = ', best_nplo)






	raise Exception('this is all you want to do right now.')



	#for i in np.arange(0,master_ppe_period_days_stack.shape[0],1):
		#plt.plot(master_ppe_period_days_stack[i], master_ppe_pttv_epochs_stack[i], alpha=0.25)
	plt.scatter(master_ppe_period_days_stack, master_ppe_pttv_epochs_stack, s=20, alpha=0.5, facecolor='DodgerBlue', edgecolor='k')

	plt.xlabel(r'$P_P$ [days]')
	plt.ylabel(r'$P_{TTV}$ [epochs]')
	plt.xscale('log')
	plt.show()





	#### MEAN AND MEDIAN LS POWER PLOT
	fig, (ax1, ax2) = plt.subplots(2, sharex=True)
	#ax1.plot(LSperiods, LSpowers, color='k', alpha=0.5, zorder=1)
	ax1.plot(LSperiods, running_mean, color='LightCoral', zorder=0)
	ax1.set_ylabel('mean power')
	ax2.set_xscale('log')

	#ax2.plot(LSperiods, LSpowers, color='k', alpha=0.5, zorder=1)
	ax2.plot(LSperiods, running_median, color='DodgerBlue', zorder=0)
	ax2.set_ylabel('median power')
	ax2.set_xlabel('period [epochs]')
	ax2.set_xscale('log')
	plt.show()


except:
	traceback.print_exc()

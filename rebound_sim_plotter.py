from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas
import traceback
import os
import time
import socket
import pickle
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
import matplotlib.cm as cm
#from customized_violin import create_violin
from violin_plotter import create_violin
from astropy.timeseries import LombScargle
from scipy.stats import skew, kurtosis
from mp_plotter import *
from astropy.constants import G, M_sun, M_jup, M_earth, R_sun, R_jup, R_earth
from sklearn.cluster import DBSCAN
from sklearn.cluster import DBSCAN

plt.rcParams["font.family"] = 'serif'


sim_prefix = input('What is the prefix of the run? (leave blank for the most recent run: ')
use_BIC_or_AIC = input("Do you want to use 'B'IC or 'A'IC? ")

if (len(sim_prefix) != 0) and os.path.exists('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/'+sim_prefix) == False:
	#### make it
	os.system('mkdir /data/tethys/Documents/Projects/NMoon_TTVs/Plots/'+sim_prefix)


if len(sim_prefix) == 0:
	sim_prefix_ = sim_prefix
else:
	if sim_prefix[-1] != '_':
		sim_prefix_ = sim_prefix+'_' #### for use in filenames... want an underscore before!

def sinewave(tvals, frequency, offset):
	return amplitude * np.sin(2*np.pi*frequency*tvals + offset)


def chisquare(data,model,error):
	return np.nansum(((data-model)**2 / (error**2)))

def BIC(nparams,data,model,error):
	return nparams*np.log(len(data)) + chisquare(data,model,error)

def AIC(nparams,data,model,error):
	return 2*nparams + chisquare(data,model,error)


def hmcontour(xvals, yvals, bins=[20,20], levels=5, colors='k'):
	##### this function will take x and y data and output contours
	##### it uses the heatmap function
	xbins, ybins = bins
	hm = plt.hist2d(xvals, yvals, bins=[xbins, ybins], cmap='coolwarm')
	hm_vals, hm_xedges, hm_yedges, hm_qm = hm 
	xcenters, ycenters, xcoords, ycoords = [], [], [], []
	for nxe,xe in enumerate(hm_xedges):
		try:	
			xcenters.append(np.nanmean((hm_xedges[nxe+1], hm_xedges[nxe])))
			xcoords.append(nxe)
		except:
			pass

	for nye,ye in enumerate(hm_yedges):
		try:
			ycenters.append(np.nanmean((hm_yedges[nye+1], hm_yedges[nye])))
			ycoords.append(nye)
		except:
			pass

	#plt.clf()
	#lt.cla()
	contplot = plt.contour(xcoords, ycoords, hm_vals, levels=levels, colors=colors)	
	plt.show()

	return contplot 






##### this script will accept a REBOUND simulation number and plot everything for that system

try:
	if socket.gethostname() == 'tethys.asiaa.sinica.edu.tw':
		#projectdir = '/data/tethys/Documents/Projects/NMoon_TTVs'
		projectdir = '/run/media/amteachey/Auddy_Akiti/Teachey/Nmoon_TTVs'
	elif socket.gethostname() == 'Alexs-MacBook-Pro.local':
		projectdir = '/Users/hal9000/Documents/Projects/Nmoon_TTVsim'
	else:
		projectdir = input('Please input the project directory: ')

	positionsdir = projectdir+'/'+sim_prefix_+'sim_positions'
	if 'RUN4' in sim_prefix:
		ttvfiledir = projectdir+'/'+sim_prefix_+'FIXED_sim_TTVs'
		LSdir = projectdir+'/'+sim_prefix_+'FIXED_sim_periodograms'
	else:
		ttvfiledir = projectdir+'/'+sim_prefix_+'sim_TTVs'
		LSdir = projectdir+'/'+sim_prefix_+'sim_periodograms'
	modeldictdir = projectdir+'/'+sim_prefix_+'sim_model_settings'
	plotdir = projectdir+'/'+sim_prefix_+'sim_plots'


	summaryfile = pandas.read_csv(projectdir+'/'+sim_prefix_+'simulation_summary.csv')
	sumfile_cols = summaryfile.columns		

	plot_individual = input('Do you want to plot an individual simulation? y/n: ')
	if plot_individual == 'y':
		sim_num = int(input('What is the simulation number you want to see? '))
		sims = np.array([sim_num])
		show_individual_plots = 'y' 
		fit_TTVs = 'y'
		cross_validate_LSPs = 'n'


	else:
		sims = np.array(summaryfile['sim'])
		show_individual_plots = input('Do you want to show individual system plots? y/n: ')
		fit_TTVs = input('Do you want to fit the TTVs (slow)? y/n: ')
		sim_obs_summary = open(projectdir+'/'+sim_prefix_+'simulated_observations.csv', mode='w')
		sim_obs_summary.write('sim,Nmoons,Pplan_days,ntransits,TTV_rmsamp_sec,TTVperiod_epochs,peak_power,fit_sineamp,deltaBIC,deltaAIC,MEGNO,SPOCK_prob\n')
		sim_obs_summary.close()

		cross_validate_LSPs = input('Do you want to do the cross-validation on PTTV periods? ')
		cv_frac_to_leave_out = 0.05
		cv_ntrials = int(1/cv_frac_to_leave_out) ### 20 trial




	#### NEW SUMMARY STATISTICS LISTS
	deltaBIC_list = []
	deltaAIC_list = []
	max_fractional_delta_rvals = []
	Msats_over_Mps = []
	Pplans_days = []
	peak_power_periods_list = []
	nmoons = np.array(summaryfile['nmoons'])
	megno_vals = np.array(summaryfile['MEGNO'])
	spockprobs = np.array(summaryfile['SPOCK_survprop'])
	try:
		Rstars = np.array(summaryfile['Rstar']) ### Rsol
	except:
		Rstars = np.linspace(1,1,len(megno_vals)) ### Rsol

	Mstars_Msol = Rstars**1.136
	Mstars_kg = Mstars_Msol * M_sun.value 



	sim_cv_results_robust_bool = []
	sim_smas_meters = []
	sim_Tdurs_hours = []




	##### compute megno and spock limits
	spock_min = 0.9 #### CONSIDER REVISING THIS!!!!! MIGHT NOT HAVE ENOUGH!!!!
	megnos_above_90 = megno_vals[spockprobs >= 0.9]
	twosig_megno_limits = np.nanpercentile(megnos_above_90, 2.5), np.nanpercentile(megnos_above_90, 97.5)
	megno_min, megno_max = twosig_megno_limits[0], twosig_megno_limits[1]


	#### DO THE MEGNO VS SPOCK PLOT HERE, COLOR CODE BY NUMBER OF MOONS
	##### AND MAKE THE CUTS DYNAMIC
	#colors = cm.Accent(np.linspace(0,1,5))
	fig = plt.figure(figsize=(6,8))
	ax = plt.subplot(111)
	colors = cm.viridis(np.linspace(0,1,5))
	#plt.plot(sim_smas_fracHill, sim_mass_ratios, color=colors[sim_nmoons-1], linewidth=2, zorder=0)
	for nmoon in np.arange(3,6,1):
		nmoon_idxs = np.where(nmoons == nmoon)[0]
		plt.scatter(megno_vals[nmoon_idxs], spockprobs[nmoon_idxs], facecolor=colors[nmoon-1], edgecolor='k', s=20, alpha=0.5, label='N = '+str(nmoon), zorder=0)
	plt.plot(np.linspace(megno_min, megno_min, 100), np.linspace(0.8,1,100), c='k', linestyle='--', alpha=0.5, linewidth=2, zorder=1)
	plt.plot(np.linspace(megno_max, megno_max, 100), np.linspace(0.8,1,100), c='k', linestyle='--', alpha=0.5, linewidth=2, zorder=1)
	plt.ylim(0.8,1)
	plt.xlim(1.4,3.0)
	plt.xlabel('MEGNO')
	plt.ylabel('SPOCK probability')
	plt.legend()
	plt.subplots_adjust(left=0.125, bottom=0.09, right=0.9, top=0.95, wspace=0.2, hspace=0.2)	
	if (sim_prefix == '') or (sim_prefix == '_'):
		plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/LAST_RUN/variable_host_spock_vs_megno.pdf', dpi=300)
		plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/LAST_RUN/variable_host_spock_vs_megno.png', dpi=300)
	else:
		plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/'+sim_prefix+'/spock_vs_megno.pdf', dpi=300)
		plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/'+sim_prefix+'/spock_vs_megno.png', dpi=300)	

	plt.show()




	if cross_validate_LSPs == 'y':
		cvpath = '/run/media/amteachey/Auddy_Akiti/Teachey/Nmoon_TTVs/'+sim_prefix+'sim_PTTV_results.csv'
		try:
			if os.path.exists(cvpath):

				### find last record KOI number:
				crossvalfile = pandas.read_csv(cvpath)
				cv_sims_examined = np.array(crossvalfile['sim']).astype(str)

			else:
				crossval_resultsfile = open(cvpath, mode='w')
				crossval_resultsfile.write('sim,n_crossval_trials,n_epochs,n_removed,PTTV_median,PTTV_std,PTTV_skew,PTTV_kurtosis,PTTV_pcterr,ATTV_median,ATTV_std,ATTV_pcterr,phase_median,phase_std,phase_pcterr,deltaBIC,deltaBIC_std,deltaAIC,deltaAIC_std\n')
				crossval_resultsfile.close()
				cv_sims_examined = np.array([])
		except:
			pass


	else:
		try:
		#cv_sims_examined = np.array([])
			crossvalfile = pandas.read_csv(cvpath)
			cv_sims_examined = np.array(crossvalfile['sim']).astype(str)
			cv_columns = crossvalfile.columns
		except:
			cv_sims_examined = np.array([])


	TTV_amplitudes = []

	for nsim,sim in enumerate(sims):


		try:

			#if (cross_validate_LSPs == 'y') and (str(sim) in cv_sims_examined):
			#	print('sim '+str(sim)+' already cross-validated. skipping.')
				#continue

			#else: #### if cross_validate_LSPs == NO OR it's not already been examined.
			


			if sim in cv_sims_examined:
			#try:
				#### grab the sim index -- it should be the same! to calculate whether the TTV detection is robust.	
				###### THESE ARE NEVER USED ANYWHERE!
				sim_cv_idx = np.where(np.array(crossvalfile['sim']).astype(str) == str(sim))[0][0]
				print('sim, nsim, sim_cv_idx = ', sim, nsim, sim_cv_idx)
				sim_cv_PTTV_pcterr = np.array(crossvalfile['PTTV_pcterr']).astype(float)[sim_cv_idx]
				sim_cv_ATTV_pcterr = np.array(crossvalfile['ATTV_pcterr']).astype(float)[sim_cv_idx]
				sim_cv_phase_pcterr = np.array(crossvalfile['phase_pcterr']).astype(float)[sim_cv_idx]
				if (sim_cv_PTTV_pcterr <= 5) and (sim_cv_ATTV_pcterr <= 5) and (sim_cv_phase_pcterr <= 5):
					sim_cv_results_robust = True
				else:
					sim_cv_results_robust = False
			else:
				pass
				"""
				#raise Exception('Could not load the cross validation results.')
				#traceback.print_exc()
				print('Something went wrong finding the sim_cv_correspondence. SKIPPING.')
				Msats_over_Mps.append(np.nan)
				sim_cv_results_robust_bool.append(np.nan)
				deltaBIC_list.append(np.nan)
				max_fractional_delta_rvals.append(np.nan) 
				peak_power_periods_list.append(np.nan)
				TTV_amplitudes.append(np.nan)
				continue
				"""
			



			print('sim # '+str(sim))
			#### MODEL INPUTS
			#dicttime1 = time.time()
			sim_model_dict = pickle.load(open(modeldictdir+'/TTVsim'+str(sim)+'_system_dictionary.pkl', "rb"))
			#dicttime2 = time.time()
			#print('dictionary load time = ', dicttime2 - dicttime1)		
			if len(sim_prefix) == 0:
				##### the new run has recorded the semimajor axis, in meters. 
				sim_sma_meters = sim_model_dict['Planet']['a']
				sim_Rstar = sim_model_dict['Planet']['Rstar'] ### meters
				sim_Pplan_days = sim_model_dict['Planet']['Pp'] ### days!
				sim_aoverRstar = sim_sma_meters / sim_Rstar
				sim_transit_theta = np.arctan(sim_Rstar / sim_sma_meters)
				sim_Tdur_days = sim_transit_theta * (sim_Pplan_days / np.pi) #### (Tdur / 2theta) = (Pplan /2pi) -- factor of 2 cancels.
				sim_Tdur_hours = sim_Tdur_days * 24 
				sim_smas_meters.append(sim_sma_meters)
				sim_Tdurs_hours.append(sim_Tdur_hours)

			else:
				#### it's not cataloged. You need to calculate it (assuming solar analog).
				sim_Pplan_seconds = sim_model_dict['Planet']['P'] ### seconds
				sim_Pplan_days = sim_Pplan_seconds / (60 * 60 * 24)
				sim_sma_meters = ((sim_Pplan_seconds**2 * G.value * (M_sun.value + M_jup.value)) / (4*np.pi**2))**(1/3)
				sim_Rstar = R_sun.value
				sim_transit_theta = np.arctan(sim_Rstar / sim_sma_meters)
				sim_Tdur_days = sim_transit_theta * (sim_Pplan_days / np.pi) #### (Tdur / 2theta) = (Pplan / 2pi) -- factor of 2 cancels.
				sim_Tdur_hours = sim_Tdur_days * 24
				sim_smas_meters.append(sim_sma_meters)
				sim_Tdurs_hours.append(sim_Tdur_hours)


			#### TTV file
			#ttvtime1 = time.time()
			sim_TTVs = pandas.read_csv(ttvfiledir+'/TTVsim'+str(sim)+'_TTVs.csv')
			sim_TTV_epochs = np.array(sim_TTVs['epoch']).astype(int)
			sim_TTV_OminusC = np.array(sim_TTVs['TTVob']).astype(float)
			sim_TTV_errors = np.array(sim_TTVs['timing_error']).astype(float)
			sim_rms = np.sqrt(np.nanmean(sim_TTV_OminusC**2))



			##### OUTLIER REJECTION #########################################################
			#DBSCAN_vector = np.vstack((KOI_epochs, KOI_OCs)).T 
			DBSCAN_vector = np.vstack((sim_TTV_epochs, sim_TTV_OminusC)).T 			
			#db = DBSCAN(eps=5*np.nanmedian(KOI_OCerrs), min_samples=int(len(KOI_epochs)/5)).fit(KOI_OCs.reshape(-1,1))			
			db = DBSCAN(eps=5*np.nanmedian(sim_TTV_errors), min_samples=int(len(sim_TTV_epochs)/5)).fit(sim_TTV_OminusC.reshape(-1,1))						
			labels = db.labels_ 
			outlier_idxs = np.where(labels == -1)[0]
			sim_TTV_epochs, sim_TTV_OminusC, sim_TTV_errors = np.delete(sim_TTV_epochs, outlier_idxs), np.delete(sim_TTV_OminusC, outlier_idxs), np.delete(sim_TTV_errors, outlier_idxs)
			KOI_rms = np.sqrt(np.nanmean(sim_TTV_OminusC**2))





			if cross_validate_LSPs == 'y':
				##### NEW FEBRUARY 2nd -- CROSS-VALIDATE PTTV RESULTS.
				#### lists to be used for evaluating the robustness of the periodogram results.
				cv_best_periods = []
				cv_deltaBICs = []
				cv_deltaAICs = []
				cv_amplitudes = []
				cv_phases = []
				cv_popts = []
				cv_pcovs = []

				ntoremove = int(cv_frac_to_leave_out*len(sim_TTV_epochs))
				if ntoremove < 1:
					ntoremove = 1
					cv_ntrials_this_time = len(sim_TTV_epochs)
				else:
					cv_ntrials_this_time = cv_ntrials

				for cv_trialnum in np.arange(0,cv_ntrials_this_time,1):
					if ntoremove == 1:
						idxs_to_leave_out = cv_trialnum ### make sure you leave out every point, one per trial
					else:
						idxs_to_leave_out = np.random.randint(low=0, high=len(sim_TTV_epochs), size=ntoremove)	
					cv_sim_epochs = np.delete(sim_TTV_epochs, idxs_to_leave_out)
					cv_sim_OCs = np.delete(sim_TTV_OminusC, idxs_to_leave_out)
					cv_sim_OCerrs = np.delete(sim_TTV_errors, idxs_to_leave_out)

					LSperiods = np.logspace(np.log10(2), np.log10(500), 5000)
					LSfreqs = 1/LSperiods
					LSpowers = LombScargle(cv_sim_epochs, cv_sim_OCs, cv_sim_OCerrs).power(LSfreqs)
					peak_power_idx = np.nanargmax(LSpowers)
					peak_power_period = LSperiods[peak_power_idx]
					peak_power_freq = 1/peak_power_period

					if cv_trialnum == 0:
						LSpowers_stack = LSpowers
					else:
						LSpowers_stack = np.vstack((LSpowers_stack, LSpowers))

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
					popt, pcov = curve_fit(sinecurve, cv_sim_epochs, cv_sim_OCs, sigma=cv_sim_OCerrs, bounds=([0, -2*np.pi], [20*sim_rms, 2*np.pi]))
					cv_popts.append(popt)
					cv_pcovs.append(pcov)
					cv_amplitudes.append(popt[0])
					cv_phases.append(popt[1])
					
					#### calculate BIC and deltaBIC -- USE ALL THE DATAPOINTS!
					#### OLD WAY BELOW 
					#BIC_flat = chisquare(data=sim_TTV_OminusC, model=np.linspace(0,0,len(sim_TTV_OminusC)),error=sim_TTV_errors) #k = 0
					#BIC_curve = 2*np.log(len(sim_TTV_OminusC)) + chisquare(data=sim_TTV_OminusC, model=sinecurve(sim_TTV_epochs, *popt), error=sim_TTV_errors) # k=2

					#### NEW WAY -- APRIL 1 2021
					#def BIC(nparams,data,model,error):
					#def AIC(nparams,data,model,error):
					BIC_flat = BIC(nparams=0, data=sim_TTV_OminusC, model=np.linspace(0,0,len(sim_TTV_OminusC)), error=sim_TTV_errors)
					BIC_curve = BIC(nparams=2, data=sim_TTV_OminusC, model=sinecurve(sim_TTV_epochs, *popt), error=sim_TTV_errors)
					AIC_flat = AIC(nparams=0, data=sim_TTV_OminusC, model=np.linspace(0,0,len(sim_TTV_OminusC)), error=sim_TTV_errors)
					AIC_curve = AIC(nparams=2, data=sim_TTV_OminusC, model=sinecurve(sim_TTV_epochs, *popt), error=sim_TTV_errors)					



					### we want BIC_curve to be SMALLER THAN BIC_flat, despite the penalty, for the SINE MODEL TO HOLD WATER.
					#### SO IF THAT'S THE CASE, AND WE DO BIC_curve - BIC_flat, then delta-BIC will be negative, which is what we want.
					deltaBIC = BIC_curve - BIC_flat 
					deltaAIC = AIC_curve - AIC_flat 

					cv_best_periods.append(peak_power_period)
					cv_deltaBICs.append(deltaBIC)
					cv_deltaAICs.append(deltaAIC)


				#### LOOK AT THE SWING ACROSS LSpowers_stack
				"""
				if show_plots == 'y':
					plt.plot(LSperiods, np.nanmedian(LSpowers_stack, axis=0), c='r')
					plt.plot(LSperiods, np.nanmedian(LSpowers_stack, axis=0)+np.nanstd(LSpowers_stack, axis=0), c='r', linestyle='--')
					plt.plot(LSperiods, np.nanmedian(LSpowers_stack, axis=0)-np.nanstd(LSpowers_stack, axis=0), c='r', linestyle='--')
					plt.xscale('log')
					plt.xlabel('Period [epochs]')
					plt.show()
				"""


				#### now compute the median and std for period fits, and the same for the deltaBIC
				cv_best_periods, cv_deltaBICs, cv_deltaAICs = np.array(cv_best_periods), np.array(cv_deltaBICs), np.array(cv_deltaAICs)
				cv_period_skew, cv_period_kurtosis = skew(cv_best_periods), kurtosis(cv_best_periods)
				cv_amplitudes, cv_phases = np.array(cv_amplitudes), np.array(cv_phases)
				cv_best_period_median, cv_best_period_std = np.nanmedian(cv_best_periods), np.nanstd(cv_best_periods)
				period_pct_error = cv_best_period_std / cv_best_period_median
				cv_deltaBICs_median, cv_deltaBICs_std = np.nanmedian(cv_deltaBICs), np.nanstd(cv_deltaBICs)
				cv_deltaAICs_median, cv_deltaAICs_std = np.nanmedian(cv_deltaAICs), np.nanstd(cv_deltaAICs)
				cv_amplitude_median, cv_amplitude_std = np.nanmedian(cv_amplitudes), np.nanstd(cv_amplitudes)
				amplitude_pct_error = cv_amplitude_std / cv_amplitude_median
				cv_phase_median, cv_phase_std = np.nanmedian(cv_phases), np.nanstd(cv_phases)
				#phase_pct_error = np.abs(cv_phase_std / cv_phase_median)
				phase_pct_error = cv_phase_std / (2*np.pi) #### SHOULD NOT BE A FRACTION OF THE VALUE! IT SHOULD BE A FRACTION OF THE CIRCLE!!!


				print('PTTV = '+str(cv_best_period_median)+' +/- '+str(cv_best_period_std))
				print('PTTV pct error = ', str(period_pct_error*100))
				print("PTTV Skew, Kurtosis = ", str(cv_period_skew), str(cv_period_kurtosis))
				print('ATTV = '+str(cv_amplitude_median)+' +/- '+str(cv_amplitude_std))
				print('ATTV pct error = ', str(amplitude_pct_error*100))
				print('Phase = '+str(cv_phase_median)+' +/- '+str(cv_phase_std))
				print('Phase pct error = ', str(phase_pct_error*100))
				print("deltaBIC = "+str(cv_deltaBICs_median)+' +/- '+str(cv_deltaBICs_std))
				print('deltaAIC = '+str(cv_deltaAICs_median)+' +/- '+str(cv_deltaAICs_std))
				print(' ')
				sim_cv_PTTV_pcterr = period_pct_error
				sim_cv_ATTV_pcterr = amplitude_pct_error
				sim_cv_phase_pcterr = phase_pct_error
				if (sim_cv_PTTV_pcterr <= 5) and (sim_cv_ATTV_pcterr <= 5) and (sim_cv_phase_pcterr <= 5):
					sim_cv_results_robust = True
				else:
					sim_cv_results_robust = False


				#if cross_validate_LSPs == 'y':
				crossval_resultsfile = open(cvpath, mode='a')
				#crossval_resultsfile.write('KOI,n_crossval_trials,n_epochs,n_removed,PTTV_median,PTTV_std,PTTV_skew,PTTV_kurtosis,PTTV_pcterr,ATTV_median,ATTV_std,ATTV_pcterr,phase_median,phase_std,phase_pcterr,deltaBIC,deltaBIC_std\n')
				crossval_resultsfile.write(str(sim)+','+str(cv_ntrials_this_time)+','+str(len(sim_TTV_epochs))+','+str(ntoremove)+','+str(cv_best_period_median)+','+str(cv_best_period_std)+','+str(cv_period_skew)+','+str(cv_period_kurtosis)+','+str(period_pct_error*100)+','+str(cv_amplitude_median)+','+str(cv_amplitude_std)+','+str(amplitude_pct_error*100)+','+str(cv_phase_median)+','+str(cv_phase_std)+','+str(phase_pct_error*100)+','+str(cv_deltaBICs_median)+','+str(cv_deltaBICs_std)+','+str(cv_deltaAICs_median)+','+str(cv_deltaAICs_std)+'\n')
				crossval_resultsfile.close()			

				print('FINISHED WITH SIM CROSS-VALIDATION.')
				#print('FINISHED WITH SIM CROSS-VALIDATION. BACK TO TOP OF LOOP. (REMOVE THIS LATER).')
				print(" ")
				#continue 




			#### PERIODOGRAM OF THE TTVs
			#periodogramtime1 = time.time()
			sim_periodogram = np.load(LSdir+'/TTVsim'+str(sim)+'_periodogram.npy')
			#periodogramtime2 = time.time()
			#print('periodogram load time = ', periodogramtime2 - periodogramtime1)


			sim_xpos = np.load(positionsdir+'/TTVsim'+str(sim)+'_xpos.npy')
			sim_ypos = np.load(positionsdir+'/TTVsim'+str(sim)+'_ypos.npy')
			sim_zpos = np.load(positionsdir+'/TTVsim'+str(sim)+'_zpos.npy')
			

			##### SUMMARY FILE ENTRY
			sim_sumfile_idx = int(np.where(np.array(summaryfile['sim']) == sim)[0])

			#### grab the summary file information
			sim_nmoons = np.array(summaryfile['nmoons'])[sim_sumfile_idx]
			sim_Pplan_days = np.array(summaryfile['Pplan_days'])[sim_sumfile_idx]
			sim_ntransits = np.array(summaryfile['ntransits'])[sim_sumfile_idx]
			sim_Msats_over_Mp = np.array(summaryfile['Mmoons_over_Mplan'])[sim_sumfile_idx]
			sim_TTV_rmsamp = np.array(summaryfile['TTV_rmsamp_sec'])[sim_sumfile_idx]
			sim_TTVperiod_epochs = np.array(summaryfile['TTVperiod_epochs'])[sim_sumfile_idx]
			sim_MEGNO = np.array(summaryfile['MEGNO'])[sim_sumfile_idx]
			sim_SPOCKprob = np.array(summaryfile['SPOCK_survprop'])[sim_sumfile_idx]


			periods, powers = sim_periodogram
			peak_power = np.nanmax(powers)
			peak_power_period = periods[np.argmax(powers)]
			#peak_power_periods_list.append(peak_power_period) #### MOVED TO THE END TO KEEP LISTS EVEN

			if show_individual_plots == 'y':
				#### plot the periodogram
				fig = plt.figure(figsize=(8,8))
				plt.plot(periods, powers, c='DodgerBlue', linewidth=2, alpha=0.7)
				plt.xlabel('Period [epochs]')
				plt.ylabel('Power')
				plt.xscale('log')
				plt.title('best period = '+str(round(peak_power_period, 3))+' epochs')
				plt.subplots_adjust(top=0.94, bottom=0.09)
				plt.show()




			#### perform a curve_fit to find amplitude and offset, to be plotted below (and for Delta-BIC -- or just delta X^2)?
			if fit_TTVs == 'y':
				frequency = 1/peak_power_period
				def sinewave(tvals, amplitude, offset):
					return amplitude * np.sin(2*np.pi*frequency*tvals + offset)

				popt, pcov = curve_fit(sinewave, sim_TTV_epochs, sim_TTV_OminusC, sigma=sim_TTV_errors, bounds=([0, -2*np.pi], [10*sim_TTV_rmsamp, 2*np.pi]))
				interp_epochs = np.linspace(np.nanmin(sim_TTV_epochs), np.nanmax(sim_TTV_epochs), 1000)
				sinecurve = sinewave(interp_epochs, *popt)
				sinecurve_amplitude = np.nanmax(np.abs(sinecurve))
				#TTV_amplitudes.append(sinecurve_amplitude)

				BIC_flat = BIC(nparams=0, data=sim_TTV_OminusC, model=np.linspace(0,0,len(sim_TTV_OminusC)), error=sim_TTV_errors)
				BIC_curve = BIC(nparams=2, data=sim_TTV_OminusC, model=sinewave(sim_TTV_epochs,*popt), error=sim_TTV_errors)
				AIC_flat = AIC(nparams=0, data=sim_TTV_OminusC, model=np.linspace(0,0,len(sim_TTV_OminusC)), error=sim_TTV_errors)
				AIC_curve = AIC(nparams=2, data=sim_TTV_OminusC, model=sinewave(sim_TTV_epochs,*popt), error=sim_TTV_errors)				

				#### we want Delta-BIC to be negative to indicate an improvement!
				###### Now, BIC_curve is an improvement over BIC_flat if BIC_curve < BIC_flat (even with extra complexity, the model is improved)
				####### so let deltaBIC = BIC_curve - BIC_flat: if BIC_curve is indeed < BIC_flat, then deltaBIC will be negative. SO:
				deltaBIC = BIC_curve - BIC_flat
				deltaAIC = AIC_curve - AIC_flat 
				#deltaBIC_list.append(deltaBIC) #### MOVED TO THE END -- SO THAT YOUR INDEXING IS OK!


				if show_individual_plots == 'y':
					#### plot the TTVs -- and fit a best fitting SINUSOID BASED ON THE PERIODOGRAM PERIOD?
					fig = plt.figure(figsize=(8,8))
					plt.scatter(sim_TTV_epochs, sim_TTV_OminusC, facecolor='LightCoral', edgecolor='k', s=20, zorder=2)
					plt.errorbar(sim_TTV_epochs, sim_TTV_OminusC, yerr=sim_TTV_errors, ecolor='k', zorder=1, fmt='none', alpha=0.5)
					plt.plot(np.linspace(np.nanmin(sim_TTV_epochs), np.nanmax(sim_TTV_epochs), 100), np.linspace(0,0,100), color='k', linestyle=':', zorder=0)
					plt.plot(interp_epochs, sinecurve, c='r', linestyle='--', linewidth=3)
					plt.xlabel('Epoch')
					plt.ylabel('O - C [s]')
					plt.title(r'$P = $'+str(round(peak_power_period,2))+r' epochs, $N_{S} = $'+str(sim_nmoons)+r', $\Delta \mathrm{BIC} = $'+str(round(deltaBIC, 2)))
					plt.subplots_adjust(top=0.94, bottom=0.09)		
					plt.show()

					###### DO A PHASE FOLD -- AND REFIT THE SIN
					phasefold_sim_TTV_epochs = sim_TTV_epochs % (peak_power_period) #### to get two cycles.
					phasefold_sort_args = np.argsort(phasefold_sim_TTV_epochs)
					phasefold_sim_TTV_OminusC_sorted = sim_TTV_OminusC[phasefold_sort_args]
					phasefold_sim_TTV_errors_sorted = sim_TTV_errors[phasefold_sort_args]
					phasefold_sim_TTV_epochs_sorted = phasefold_sim_TTV_epochs[phasefold_sort_args]
					phasefold_interp_epochs = np.linspace(np.nanmin(phasefold_sim_TTV_epochs_sorted), np.nanmax(phasefold_sim_TTV_epochs_sorted), 1000)
					phasefold_sinecurve = sinewave(phasefold_interp_epochs, *popt)
					phasefold_sinecurve_amplitude = np.nanmax(np.abs(phasefold_sinecurve))


					fig = plt.figure(figsize=(8,8))
					plt.scatter(phasefold_sim_TTV_epochs_sorted, phasefold_sim_TTV_OminusC_sorted, facecolor='LightCoral', edgecolor='k', s=20, zorder=2)
					plt.errorbar(phasefold_sim_TTV_epochs_sorted, phasefold_sim_TTV_OminusC_sorted, yerr=phasefold_sim_TTV_errors_sorted, ecolor='k', zorder=1, fmt='none', alpha=0.5)
					plt.plot(np.linspace(np.nanmin(phasefold_sim_TTV_epochs_sorted), np.nanmax(phasefold_sim_TTV_epochs_sorted), 100), np.linspace(0,0,100), color='k', linestyle=':', zorder=0)
					plt.plot(phasefold_interp_epochs, phasefold_sinecurve, c='r', linestyle='--', linewidth=3)
					#plt.xlabel('Epoch')
					plt.xlabel('epochs [phase fold]')
					plt.ylabel('O - C [s]')
					plt.title(r'$P = $'+str(round(peak_power_period,2))+r' epochs, $N_{S} = $'+str(sim_nmoons)+r', $\Delta \mathrm{BIC} = $'+str(round(deltaBIC, 2)))
					plt.subplots_adjust(top=0.94, bottom=0.09)
					plt.show()					






			fractional_delta_rvals = []
			eccentricity_estimates = [] #### will not be exactly eccentricity, since you're not measuring a single orbit's apoapse and periapse.
			for i in np.arange(0,sim_xpos.shape[0], 1): 
				tvals = np.linspace(0,10,10000) 
				rvals = np.sqrt((sim_xpos[i]**2) + (sim_ypos[i]**2)) 
				min_rval, max_rval = np.nanmin(rvals), np.nanmax(rvals)
				delta_r = max_rval - min_rval
				fractional_delta_r = delta_r / np.nanmean((min_rval, max_rval))
				fractional_delta_rvals.append(fractional_delta_r)
				eccentricity_swing = (max_rval - min_rval) / (max_rval + min_rval)

				"""
				if show_individual_plots == 'y':
					plt.plot(tvals, rvals)
					if i != 0:
						print('Moon # '+str(i))
						print('fractional Delta-semimajor axis: ', fractional_delta_r)
						print('eccentricity swing: ', eccentricity_swing)
						print(' ')
					plt.xlabel('time')
					plt.ylabel(r'$R_{CoM}$')
					plt.title("Moon "+str(i)+", MEGNO = "+str(round(sim_MEGNO,2))+r', $P_{\mathrm{spock}} = $'+str(round(sim_SPOCKprob,2)))
				plt.show()    
				"""
			maximum_fractional_delta_r = np.nanmax(fractional_delta_rvals)

			#max_fractional_delta_rvals.append(maximum_fractional_delta_r)  ### MOVED TO THE END, TO KEEP LISTS EVEN



			#### plot the positions
			colors = cm.get_cmap('viridis')(np.linspace(0,1,6))
			nparticles = sim_xpos.shape[0]
			if show_individual_plots == 'y':
				fig = plt.figure(figsize=(8,8))
				#fig, ax = plt.figure(figsize=(6,6))
				for particle in np.arange(0,nparticles,1):
					part_xpos, part_ypos = sim_xpos[particle], sim_ypos[particle]
					plt.plot(part_xpos, part_ypos, color=colors[particle], alpha=0.7)
				plt.xlabel(r'$r \, / \, a_{I}$')
				plt.ylabel(r'$r \, / \, a_{I}$')
				plt.title("MEGNO = "+str(round(sim_MEGNO, 2))+', stability prob = '+str(round(sim_SPOCKprob*100,2))+'%')
				plt.subplots_adjust(top=0.94, bottom=0.09)						
				plt.show()


			#### WRITE OUT THIS INFORMATION TO THE SIMULATION OBSERVATIONS SUMMARY FILE !!!! THESE CAN BEN ANN INPUTS.
			if plot_individual == 'n':
				sim_obs_summary = open(projectdir+'/'+sim_prefix_+'simulated_observations.csv', mode='a')		
				#sim_obs_summary.write('sim,Pplan_days,ntransits,	TTV_rmsamp_sec,TTVperiod_epochs,peak_power,fit_sineamp,deltaBIC,MEGNO,SPOCK_prob\n')
				sim_obs_summary.write(str(sim)+','+str(sim_nmoons)+','+str(sim_Pplan_days)+','+str(sim_ntransits)+','+str(sim_TTV_rmsamp)+','+str(sim_TTVperiod_epochs)+','+str(peak_power)+','+str(popt[0])+','+str(deltaBIC)+','+str(deltaAIC)+','+str(sim_MEGNO)+','+str(sim_SPOCKprob)+'\n')
				sim_obs_summary.close()


			#### PLACE MISCELANEOUS LIST APPENDS DOWN HERE.
			Msats_over_Mps.append(sim_Msats_over_Mp)
			Pplans_days.append(sim_Pplan_days)
			try:
				sim_cv_results_robust_bool.append(sim_cv_results_robust)
			except:
				sim_cv_results_robust_bool.append(np.nan)
			deltaBIC_list.append(deltaBIC) #### MOVED TO THE END -- SO THAT YOUR INDEXING IS OK!
			deltaAIC_list.append(deltaAIC)
			max_fractional_delta_rvals.append(maximum_fractional_delta_r)  ### MOVED TO THE END, TO KEEP LISTS EVEN
			peak_power_periods_list.append(peak_power_period)
			TTV_amplitudes.append(sinecurve_amplitude)

		except:
			#### APPENDING NaNs so we can keep all the lists the same length!
			Msats_over_Mps.append(np.nan)
			Pplans_days.append(np.nan)
			sim_cv_results_robust_bool.append(np.nan)
			deltaBIC_list.append(np.nan)
			deltaAIC_list.append(np.nan)
			max_fractional_delta_rvals.append(np.nan) 
			peak_power_periods_list.append(np.nan)
			TTV_amplitudes.append(np.nan)

			traceback.print_exc()
			continue 



	#### MAKE THE LISTS INTO ARRAYS
	deltaBIC_list = np.array(deltaBIC_list)
	deltaAIC_list = np.array(deltaAIC_list)
	peak_power_periods_list = np.array(peak_power_periods_list)
	max_fractional_delta_rvals = np.array(max_fractional_delta_rvals)
	Msats_over_Mps = np.array(Msats_over_Mps)
	Pplans_days = np.array(Pplans_days)
	sim_cv_results_robust_bool = np.array(sim_cv_results_robust_bool)
	sim_Tdurs_hours = np.array(sim_Tdurs_hours)








	##### DATA CUTS
	stable_megno_idxs = np.where((megno_vals >= megno_min) & (megno_vals <= megno_max))[0]
	stable_spockprobs = np.where(spockprobs >= spock_min)[0] #### UPDATE BASED ON DEARTH OF P > 0.9 systems in the new paradigm.
	cv_robust_idxs = np.where(sim_cv_results_robust_bool == True)[0] #### CROSS-VALIDATION IS ROBUST -- <= 5% error on period, amplitude, and phase.
	unstable_megno_idxs = np.concatenate((np.where(megno_vals < megno_min)[0], np.where(megno_vals > megno_max)[0]))
	unstable_spockprobs = np.where(spockprobs < spock_min)[0]
	bad_PTTV_solution_idxs = np.where(peak_power_periods_list == 2.0)[0] #### if you're right on 2.0, it's not reliable solution.
	good_PTTV_solution_idxs = []
	for nppp, ppp in enumerate(peak_power_periods_list):
		if nppp not in bad_PTTV_solution_idxs:
			good_PTTV_solution_idxs.append(nppp)
	good_PTTV_solution_idxs = np.array(good_PTTV_solution_idxs)
	stable_bool = []
	stable_idxs = []
	unstable_idxs = []

	#### STABILITY CHECK
	for idx in np.arange(0,len(megno_vals),1):
		if (idx in stable_spockprobs):
			stable_idxs.append(idx) #### if SPOCK probability is good, we go with this
			stable_bool.append(True)
		
		else:
			#### it's not in stable_spockprobs
			if np.isfinite(spockprobs[idx]) == True:
				#### it's present, just not good enough
				unstable_idxs.append(idx)
				stable_bool.append(False)

			else:
				if idx in stable_megno_idxs:
					stable_idxs.append(idx)
					stable_bool.append(True)
				else:
					unstable_idxs.append(idx)
					stable_bool.append(False)



	stable_idxs = np.array(stable_idxs)		
	unstable_idxs = np.array(unstable_idxs)
	stable_bool = np.array(stable_bool)

	#### call it right here -- stable idxs should ALSO be P != 2.0 EXACTLY.
	original_stable_idxs = [] #### BEFORE THE CUT BELOW!!!!!
	for sidx in stable_idxs:
		original_stable_idxs.append(sidx)
	stable_idxs = np.intersect1d(stable_idxs, good_PTTV_solution_idxs) #### THIS SHOULD PROPAGATE BELOW.


	try:
		#### CREATE A VIOLIN PLOT OF deltaBICs as a function of moons
		violin_data = [[], [], [], [], []] #### a list of lists
		for moonidx, nmoon in enumerate(np.arange(1,6,1)):
			nmoon_idxs = np.where(nmoons == nmoon)[0]
			nmoon_stable_idxs = np.intersect1d(stable_idxs, nmoon_idxs)
			nmoon_good_deltaBIC_idxs = np.intersect1d(np.where(nmoons == nmoon)[0], np.where(deltaBIC_list <= -2)[0])
			nmoon_good_deltaBIC_and_stable_idxs = np.intersect1d(nmoon_stable_idxs, nmoon_good_deltaBIC_idxs)
			nmoon_deltaBICs = deltaBIC_list[nmoon_stable_idxs] #### WANT TO SHOW ALL! NOT JUST GOOD.
			nmoon_good_deltaAIC_idxs = np.intersect1d(np.where(nmoons == nmoon)[0], np.where(deltaAIC_list <= 0)[0]) #### don't know a good cut-off for AIC, just use negative (improvement)
			nmoon_good_deltaAIC_and_stable_idxs = np.intersect1d(nmoon_stable_idxs, nmoon_good_deltaAIC_idxs)
			nmoon_deltaAICs = deltaAIC_list[nmoon_stable_idxs] #### WANT TO SHOW ALL! NOT JUST GOOD.			


			violin_data[moonidx] = nmoon_deltaBICs
			print('nmoon = ', nmoon)
			print('total available: ', len(nmoon_idxs))
			print('number stable: ', len(nmoon_stable_idxs))
			print('pct stable: ', (len(nmoon_stable_idxs) / len(nmoon_idxs))*100)
			print('number w/ good deltaBIC: ', len(np.where(nmoon_deltaBICs <= -2)[0]))
			print('pct w/ good deltaBIC: ', (len(np.where(nmoon_deltaBICs <= -2)[0]) / len(nmoon_idxs))*100)
			print('number stable AND good deltaBIC: ', len(nmoon_good_deltaBIC_and_stable_idxs))
			print('pct stable AND good deltaBIC: ', (len(nmoon_good_deltaBIC_and_stable_idxs) / len(nmoon_idxs))*100)
			print('number w/ good deltaAIC: ', len(np.where(nmoon_deltaAICs <= 0)[0]))
			print('pct w/ good deltaAIC: ', (len(np.where(nmoon_deltaAICs <= 0)[0]) / len(nmoon_idxs))*100)
			print('number stable AND good deltaAIC: ', len(nmoon_good_deltaAIC_and_stable_idxs))
			print('pct stable AND good deltaAIC: ', (len(nmoon_good_deltaAIC_and_stable_idxs) / len(nmoon_idxs))*100)

			print(' ')
		#create_violin(data, data_labels=None, x_label=None, y_label=None, plot_title=None, colormap='viridis')
		create_violin(violin_data, data_labels=['1', '2', '3', '4','5'], x_label='# moons', y_label=r'$\Delta$ BIC', colormap='viridis', autoshow=False)
		plt.plot(np.linspace(0,6,100), np.linspace(-2,-2,100), c='k', linestyle=':', alpha=0.7)
		plt.show()
	except:
		print('could not produce the violin plot.')


	try:


		#### create five histograms of the deltaBICS
		colors = cm.get_cmap('viridis')(np.linspace(0,1,5))
		color_idxs = np.linspace(0,1,5)
		fig, ax = plt.subplots(5, sharex=True, figsize=(6,8))
		nmoon_x2_histbins = np.linspace(-100,10,50)
		
		for moonidx, nmoon in enumerate(np.arange(1,6,1)):
			nmoon_idxs = np.where(nmoons == nmoon)[0]
			nmoon_stable_idxs = np.intersect1d(stable_idxs, nmoon_idxs)
			nmoon_deltaBICs = deltaBIC_list[nmoon_stable_idxs]
			nmoon_good_deltaBIC_idxs = np.intersect1d(np.where(nmoons == nmoon)[0], np.where(deltaBIC_list <= -2)[0])
			nmoon_good_deltaBIC_and_stable_idxs = np.intersect1d(nmoon_stable_idxs, nmoon_good_deltaBIC_idxs)

			nmoon_median_deltaBIC = np.nanmedian(nmoon_deltaBICs)

			nmoon_deltaAICs = deltaAIC_list[nmoon_stable_idxs]
			nmoon_good_deltaAIC_idxs = np.intersect1d(np.where(nmoons == nmoon)[0], np.where(deltaAIC_list <= 0)[0])
			nmoon_good_deltaAIC_and_stable_idxs = np.intersect1d(nmoon_stable_idxs, nmoon_good_deltaAIC_idxs)

			nmoon_median_deltaAIC = np.nanmedian(nmoon_deltaAICs)			

			print('nmoon = ', nmoon)
			print('total available: ', len(nmoon_idxs))
			print('number stable: ', len(nmoon_stable_idxs))
			print('pct stable: ', (len(nmoon_stable_idxs) / len(nmoon_idxs))*100)
			print('number w/ good deltaBIC: ', len(np.where(nmoon_deltaBICs <= -2)[0]))
			print('pct w/ good deltaBIC: ', (len(np.where(nmoon_deltaBICs <= -2)[0]) / len(nmoon_idxs))*100)
			print('number stable AND good deltaBIC: ', len(nmoon_good_deltaBIC_and_stable_idxs))
			print('pct stable AND good deltaBIC: ', (len(nmoon_good_deltaBIC_and_stable_idxs) / len(nmoon_idxs))*100)

			print('number w/ good deltaAIC: ', len(np.where(nmoon_deltaAICs <= 0)[0]))
			print('pct w/ good deltaAIC: ', (len(np.where(nmoon_deltaAICs <= 0)[0]) / len(nmoon_idxs))*100)
			print('number stable AND good deltaAIC: ', len(nmoon_good_deltaAIC_and_stable_idxs))
			print('pct stable AND good deltaAIC: ', (len(nmoon_good_deltaAIC_and_stable_idxs) / len(nmoon_idxs))*100)			
			print(' ')
			axn,axb,axe = ax[moonidx].hist(nmoon_deltaBICs, bins=nmoon_x2_histbins, facecolor=colors[moonidx], alpha=0.7, edgecolor='k')
			ax[moonidx].set_ylabel('N = '+str(nmoon))
			ax[moonidx].plot(np.linspace(nmoon_median_deltaBIC, nmoon_median_deltaBIC, 100), np.linspace(0,1.05*np.nanmax(axn),100), c='k', linestyle=':')
			ax[moonidx].set_xlim(-100,10)
			ax[moonidx].set_ylim(0,1.05*np.nanmax(axn))
		ax[4].set_xlabel(r'$\Delta \, \mathrm{BIC}$')

		plt.subplots_adjust(left=0.15, bottom=0.1, right=0.85, top=0.95, wspace=0.2, hspace=0.1)
		if (sim_prefix == '') or (sim_prefix == '_'):
			plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/LAST_RUN/variable_host_deltaBIC_histogram_all.pdf', dpi=300)			
			plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/LAST_RUN/variable_host_deltaBIC_histogram_all.png', dpi=300)	
		else:
			plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/'+sim_prefix+'/deltaBIC_histogram_all.pdf', dpi=300)
			plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/'+sim_prefix+'/deltaBIC_histogram_all.png', dpi=300)				

		plt.show()
	except:
		print('could not produce the deltaBIC histogram plot.')	





	try:
		#### create five histograms of the deltaBICS -- WEIGHTED BY GEOMETRIC TRANSIT PROBABILITY!
		colors = cm.get_cmap('viridis')(np.linspace(0,1,5))
		color_idxs = np.linspace(0,1,5)
		fig, ax = plt.subplots(5, sharex=True, figsize=(6,8))
		nmoon_x2_histbins = np.linspace(-60,10,50)
		
		shortest_period_days = np.nanmin(np.array(summaryfile['Pplan_days']))
		shortest_period_seconds = shortest_period_days * 24 * 60 * 60
		shortest_period_idx = np.nanargmin(np.array(summaryfile['Pplan_days']))
		shortest_period_mstar_kg = Mstars_kg[shortest_period_idx]
		shortest_period_sma_meters = ((shortest_period_seconds**2 * G.value * shortest_period_mstar_kg) / (4*np.pi**2))**(1/3)


		#inclusion_weight_denominators = np.array(summaryfile['Pplan_days'])/shortest_period
		inclusion_weight_denominators = np.array(sim_smas_meters) / shortest_period_sma_meters
		#### JUSTIFICATION: tan(theta) = Rstar / aplan approximately = Rstar / aplan. Twice the aplan, halve the theta, halve the transit probability.
		#### WE'RE SCALING THESE PROBABILITIES SUCH THAT smallest sma has probability = 1. If you have twice the sma of this value, your inclusion probability will be 50%.
		scaled_transit_probabilities = 1/inclusion_weight_denominators 



		for moonidx, nmoon in enumerate(np.arange(1,6,1)):
			nmoon_idxs = np.where(nmoons == nmoon)[0]
			nmoon_stable_idxs = np.intersect1d(stable_idxs, nmoon_idxs)
			nmoon_deltaBICs = deltaBIC_list[nmoon_stable_idxs]
			nmoon_deltaAICs = deltaAIC_list[nmoon_stable_idxs]
			nmoon_periods = np.array(summaryfile['Pplan_days'])[nmoon_stable_idxs]
			nmoon_transit_probs = scaled_transit_probabilities[nmoon_stable_idxs] #### we've just scaled these probabilities so that the closest in planet has transit probability of one.
			##### DOING THIS TO INCREASE THE NUMBER OF SYSTEMS IN THE SAME -- NOT JUST THROWING AWAY A HUGE PERCENTAGE OF SYSTEMS WITH LOW TRANSIT PROBABILITIES.
			"""
			try:
				nmoon_sma_meters = sim_smas_meters[nmoon_stable_idxs]
				nmoon_thetas = np.arctan(R_sun.value / nmoon_sma_meters)
				nmoon_transit_prob = nmoon_thetas / (np.pi) #### the total angle is 2theta, divided by 2pi, simplifies.
			except:
				nmoon_transit_prob = inclusion_weights[nmoon_stable_idxs]
			"""


			#### LET MINIMUM PERIOD BE THE BASELINE (PROBABILITY = 1)
			##### AT TWICE THAT PERIOD, INCLUSION PROBABILITY SHOULD BE 1/2.
			##### SO IF WE DIVIDE EVERY PERIOD BY THE SHORTEST PERIOD, THAT IS THE DENOMINATOR IN THE WEIGHT FOR LIKELIHOOD OF INCLUSION.

			nmoon_final_idxs = []
			#for nmidx, nmiw in zip(nmoon_stable_idxs, nmoon_inclusion_weights):
			for nmidx, ntp in zip(nmoon_stable_idxs, nmoon_transit_probs):
				#### draw a random number -- if better than nmoon_inclusion weights, include it.
				random_draw = np.random.random()
				if ntp >= random_draw:
					nmoon_final_idxs.append(nmidx)
			nmoon_deltaBICs = deltaBIC_list[nmoon_final_idxs]
			nmoon_deltaAICs = deltaAIC_list[nmoon_final_idxs]
			nmoon_median_deltaBIC = np.nanmedian(nmoon_deltaBICs)
			nmoon_median_deltaAIC = np.nanmedian(nmoon_deltaAICs)

			axn,axb,axe = ax[moonidx].hist(nmoon_deltaBICs, bins=nmoon_x2_histbins, facecolor=colors[moonidx], alpha=0.7, edgecolor='k')
			ax[moonidx].set_ylabel('N = '+str(nmoon))
			ax[moonidx].plot(np.linspace(nmoon_median_deltaBIC, nmoon_median_deltaBIC, 100), np.linspace(0,1.05*np.nanmax(axn),100), c='r', linestyle='--', linewidth=2)
			ax[moonidx].set_xlim(-40,10)
			ax[moonidx].set_ylim(0,1.05*np.nanmax(axn))
			ax[moonidx].set_rasterization_zorder(0)
		ax[4].set_xlabel(r'$\Delta \, \mathrm{BIC}$')

		plt.subplots_adjust(left=0.15, bottom=0.1, right=0.85, top=0.95, wspace=0.2, hspace=0.1)
		if (sim_prefix == '') or (sim_prefix == '_'):
			plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/LAST_RUN/variable_host_deltaBIC_histogram_geometric_transit_bias.pdf', dpi=300)			
			plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/LAST_RUN/variable_host_deltaBIC_histogram_geometric_transit_bias.png', dpi=300)	
		else:
			plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/'+sim_prefix+'/deltaBIC_histogram_geometric_transit_bias.pdf', dpi=300)
			plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/'+sim_prefix+'/deltaBIC_histogram_geometric_transit_bias.png', dpi=300)							

		plt.show()
	except:
		print('could not produce the deltaBIC histogram plot.')	




	####### DO THE SAME NOW WITH DELTA-AIC
	try:
		#### create five histograms of the deltaBICS -- WEIGHTED BY GEOMETRIC TRANSIT PROBABILITY!
		colors = cm.get_cmap('viridis')(np.linspace(0,1,5))
		color_idxs = np.linspace(0,1,5)
		fig, ax = plt.subplots(5, sharex=True, figsize=(6,8))
		nmoon_x2_histbins = np.linspace(-60,10,50)
		
		shortest_period_days = np.nanmin(np.array(summaryfile['Pplan_days']))
		shortest_period_seconds = shortest_period_days * 24 * 60 * 60
		shortest_period_idx = np.nanargmin(np.array(summaryfile['Pplan_days']))
		shortest_period_mstar_kg = Mstars_kg[shortest_period_idx]
		shortest_period_sma_meters = ((shortest_period_seconds**2 * G.value * shortest_period_mstar_kg) / (4*np.pi**2))**(1/3)


		#inclusion_weight_denominators = np.array(summaryfile['Pplan_days'])/shortest_period
		inclusion_weight_denominators = np.array(sim_smas_meters) / shortest_period_sma_meters
		#### JUSTIFICATION: tan(theta) = Rstar / aplan approximately = Rstar / aplan. Twice the aplan, halve the theta, halve the transit probability.
		#### WE'RE SCALING THESE PROBABILITIES SUCH THAT smallest sma has probability = 1. If you have twice the sma of this value, your inclusion probability will be 50%.
		scaled_transit_probabilities = 1/inclusion_weight_denominators 



		for moonidx, nmoon in enumerate(np.arange(1,6,1)):
			nmoon_idxs = np.where(nmoons == nmoon)[0]
			nmoon_stable_idxs = np.intersect1d(stable_idxs, nmoon_idxs)
			nmoon_deltaBICs = deltaBIC_list[nmoon_stable_idxs]
			nmoon_deltaAICs = deltaAIC_list[nmoon_stable_idxs]
			nmoon_periods = np.array(summaryfile['Pplan_days'])[nmoon_stable_idxs]
			nmoon_transit_probs = scaled_transit_probabilities[nmoon_stable_idxs] #### we've just scaled these probabilities so that the closest in planet has transit probability of one.
			##### DOING THIS TO INCREASE THE NUMBER OF SYSTEMS IN THE SAME -- NOT JUST THROWING AWAY A HUGE PERCENTAGE OF SYSTEMS WITH LOW TRANSIT PROBABILITIES.
			"""
			try:
				nmoon_sma_meters = sim_smas_meters[nmoon_stable_idxs]
				nmoon_thetas = np.arctan(R_sun.value / nmoon_sma_meters)
				nmoon_transit_prob = nmoon_thetas / (np.pi) #### the total angle is 2theta, divided by 2pi, simplifies.
			except:
				nmoon_transit_prob = inclusion_weights[nmoon_stable_idxs]
			"""


			#### LET MINIMUM PERIOD BE THE BASELINE (PROBABILITY = 1)
			##### AT TWICE THAT PERIOD, INCLUSION PROBABILITY SHOULD BE 1/2.
			##### SO IF WE DIVIDE EVERY PERIOD BY THE SHORTEST PERIOD, THAT IS THE DENOMINATOR IN THE WEIGHT FOR LIKELIHOOD OF INCLUSION.

			nmoon_final_idxs = []
			#for nmidx, nmiw in zip(nmoon_stable_idxs, nmoon_inclusion_weights):
			for nmidx, ntp in zip(nmoon_stable_idxs, nmoon_transit_probs):
				#### draw a random number -- if better than nmoon_inclusion weights, include it.
				random_draw = np.random.random()
				if ntp >= random_draw:
					nmoon_final_idxs.append(nmidx)
			nmoon_deltaBICs = deltaBIC_list[nmoon_final_idxs]
			nmoon_deltaAICs = deltaAIC_list[nmoon_final_idxs]
			nmoon_median_deltaBIC = np.nanmedian(nmoon_deltaBICs)
			nmoon_median_deltaAIC = np.nanmedian(nmoon_deltaAICs)

			axn,axb,axe = ax[moonidx].hist(nmoon_deltaAICs, bins=nmoon_x2_histbins, facecolor=colors[moonidx], alpha=0.7, edgecolor='k')
			ax[moonidx].set_ylabel('N = '+str(nmoon))
			ax[moonidx].plot(np.linspace(nmoon_median_deltaAIC, nmoon_median_deltaAIC, 100), np.linspace(0,1.05*np.nanmax(axn),100), c='r', linestyle='--', linewidth=2)
			ax[moonidx].set_xlim(-40,10)
			ax[moonidx].set_ylim(0,1.05*np.nanmax(axn))
			ax[moonidx].set_rasterization_zorder(0)
		ax[4].set_xlabel(r'$\Delta \, \mathrm{AIC}$')

		plt.subplots_adjust(left=0.15, bottom=0.1, right=0.85, top=0.95, wspace=0.2, hspace=0.1)
		if (sim_prefix == '') or (sim_prefix == '_'):
			plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/LAST_RUN/variable_host_deltaAIC_histogram_geometric_transit_bias.pdf', dpi=300)			
			plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/LAST_RUN/variable_host_deltaAIC_histogram_geometric_transit_bias.png', dpi=300)	
		else:
			plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/'+sim_prefix+'/deltaAIC_histogram_geometric_transit_bias.pdf', dpi=300)
			plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/'+sim_prefix+'/deltaAIC_histogram_geometric_transit_bias.png', dpi=300)							

		plt.show()
	except:
		print('could not produce the deltaAIC histogram plot.')		













	##### make stable and unstable heatmaps, and then divide stable by total
	#nmoons_masses_heatmap = np.zeros(shape=(20,5))
	mass_bin_edges = np.logspace(-5,-2,20) #### make as many bins as possible, while keeping them all populated.
	#mass_bin_edges = np.linspace(1e-5,1e-2,20)
	moon_bin_edges = np.linspace(0.5,5.5,6) ### on the halves, because n=1 needs to go in the middle of the bin.
	stable_heatmap = plt.hist2d(Msats_over_Mps[stable_idxs], nmoons[stable_idxs], bins=[mass_bin_edges, moon_bin_edges], cmap='coolwarm')[0]
	plt.colorbar()
	plt.title('All stable')
	plt.xscale('log')
	plt.show()

	unstable_heatmap = plt.hist2d(Msats_over_Mps[unstable_idxs], nmoons[unstable_idxs], bins=[mass_bin_edges, moon_bin_edges], cmap='coolwarm')[0]	
	plt.title('all unstable')
	plt.xscale('log')
	plt.colorbar()
	plt.show()

	stable_over_total = stable_heatmap / (stable_heatmap + unstable_heatmap)
	fig = plt.figure(figsize=(6,8))
	ax = plt.subplot(111)
	plt.imshow(stable_over_total.T, origin='lower', cmap='coolwarm', aspect='auto')
	#plt.xticks(ticks=np.linspace(-0.5,18.5,3), labels=['1e-5', '5e-3', '1e-2'])
	plt.xticks(ticks=np.linspace(-0.5,18.5,5), labels=np.linspace(-5,-2,5))
	plt.yticks(ticks=[0,1,2,3,4], labels=[1,2,3,4,5])
	#plt.xlabel(r'$\log_{10} \, (\Sigma M_{\mathrm{S}}) / M_{\mathrm{P}}$')
	plt.xlabel(r'$\log_{10} \, (\Sigma M_{\mathrm{S}}) / M_{\mathrm{P}}$')
	plt.ylabel('# moons')
	#plt.xscale('log')
	#plt.yscale('log')
	cbar = plt.colorbar()
	cbar.set_label('fraction stable')
	plt.subplots_adjust(left=0.093, bottom=0.1, right=0.98, top=0.9, wspace=0.2, hspace=0.2)
	#plt.title('Fraction stable')
	
	if (sim_prefix == '') or (sim_prefix == '_'):
		plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/LAST_RUN/variable_host_stability_heatmap.pdf', dpi=300)		
		plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/LAST_RUN/variable_host_stability_heatmap.png', dpi=300)		
	else:
		plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/'+sim_prefix+'/stability_heatmap.pdf', dpi=300)
		plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/'+sim_prefix+'/stability_heatmap.png', dpi=300)			

	plt.show()





	##### DO THE SAME THING AS ABOVE, BUT PLOT IT AS A FUNCTION OF SEMIMAJOR AXIS, NOT MASS RATIO.
	##### make stable and unstable heatmaps, and then divide stable by total
	mass_bin_edges = np.logspace(-5,-2,20) #### make as many bins as possible, while keeping them all populated.
	Pplan_bin_edges = np.logspace(1,np.log10(1500),20)
	moon_bin_edges = np.linspace(0.5,5.5,6) ### on the halves, because n=1 needs to go in the middle of the bin.
	
	stable_heatmap = plt.hist2d(Pplans_days[stable_idxs], nmoons[stable_idxs], bins=[Pplan_bin_edges, moon_bin_edges], cmap='coolwarm')[0]
	plt.colorbar()
	plt.title('All stable')
	plt.xlabel(r'$P_{\mathrm{P}}$ [days]')
	plt.ylabel('# moons')
	plt.xscale('log')
	plt.show()

	unstable_heatmap = plt.hist2d(Pplans_days[unstable_idxs], nmoons[unstable_idxs], bins=[Pplan_bin_edges, moon_bin_edges], cmap='coolwarm')[0]	
	plt.title('all unstable')
	plt.xscale('log')
	plt.xlabel(r'$P_{\mathrm{P}}$ [days]')
	plt.ylabel('# moons')
	plt.colorbar()
	plt.show()

	stable_over_total = stable_heatmap / (stable_heatmap + unstable_heatmap)
	fig = plt.figure(figsize=(6,8))
	ax = plt.subplot(111)
	plt.imshow(stable_over_total.T, origin='lower', cmap='coolwarm', aspect='auto')
	#plt.xticks(ticks=np.linspace(-0.5,18.5,3), labels=['1e-5', '5e-3', '1e-2'])
	#plt.xticks(ticks=np.linspace(-0.5,18.5,5), labels=np.linspace(-5,-2,5))
	plt.xticks(ticks=np.linspace(-0.5,18.5,5), labels=np.around(np.linspace(np.log10(10), np.log10(1500),5),2))
	plt.yticks(ticks=[0,1,2,3,4], labels=[1,2,3,4,5])
	#plt.xlabel(r'$\log_{10} \, (\Sigma M_{\mathrm{S}}) / M_{\mathrm{P}}$')
	#plt.xlabel(r'$\log_{10} \, (\Sigma M_{\mathrm{S}}) / M_{\mathrm{P}}$')
	plt.xlabel(r'$\log_{10} \, P_{\mathrm{P}}$ [days]')
	plt.ylabel('# moons')
	#plt.xscale('log')
	#plt.yscale('log')
	cbar = plt.colorbar()
	cbar.set_label('fraction stable')
	plt.subplots_adjust(left=0.093, bottom=0.1, right=0.98, top=0.9, wspace=0.2, hspace=0.2)
	#plt.title('Fraction stable')
	
	if (sim_prefix == '') or (sim_prefix == '_'):
		plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/LAST_RUN/variable_host_stability_vs_Pplan_heatmap.pdf', dpi=300)		
		plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/LAST_RUN/variable_host_stability_vs_Pplan_heatmap.png', dpi=300)		
	else:
		plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/'+sim_prefix+'/stability_vs_Pplan_heatmap.pdf', dpi=300)
		plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/'+sim_prefix+'/stability_vs_Pplan_heatmap.png', dpi=300)			

	plt.show()









	try:
		#### PLOT DeltaBIC as a function of total moon mass, color coded by nmoons
		colors = cm.get_cmap('viridis')(np.linspace(0,1,5))
		color_idxs = np.linspace(0,1,5)

		fig = plt.figure(figsize=(6,8))
		ax = plt.subplot(111)

		for moonidx, nmoon in enumerate(np.arange(1,6,1)):
			nmoon_idxs = np.where(nmoons == nmoon)[0]
			nmoon_stable_idxs = np.intersect1d(stable_idxs, nmoon_idxs)
			nmoon_stable_idxs = np.intersect1d(nmoon_stable_idxs, good_PTTV_solution_idxs) #### purge all PTTV == 2.0
			nmoon_deltaBICs = deltaBIC_list[nmoon_stable_idxs]
			nmoon_deltaAICs = deltaAIC_list[nmoon_stable_idxs]
			nmoon_total_masses = Msats_over_Mps[nmoon_stable_idxs]

			plt.scatter(nmoon_total_masses, nmoon_deltaBICs, facecolor=colors[moonidx], edgecolor='k', alpha=0.5, s=20, label='N = '+str(nmoon), zorder=0)
		plt.plot(np.logspace(-5,-2,100), np.linspace(-2,-2,100), c='r', alpha=0.5, linestyle='--', linewidth=3, zorder=5)

		plt.xlabel(r'$(\Sigma \, M_S) / M_P$')
		plt.xlim(1e-5,1e-2)
		plt.ylim(-1000, 20)
		plt.xscale('log')
		#plt.yscale('log')
		plt.ylabel(r'$\Delta$ BIC')
		plt.legend(loc=3)
		ax.set_rasterization_zorder(0)
		plt.subplots_adjust(left=0.15, bottom=0.09, right=0.85, top=0.95, wspace=0.2, hspace=0.2)
		if (sim_prefix == '') or (sim_prefix == '_'):
			plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/LAST_RUN/variable_host_deltaBIC_vs_total_moon_mass.pdf', dpi=300)		
			plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/LAST_RUN/variable_host_deltaBIC_vs_total_moon_mass.png', dpi=300)		
		else:
			plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/'+sim_prefix+'/deltaBIC_vs_total_moon_mass.pdf', dpi=300)
			plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/'+sim_prefix+'/deltaBIC_vs_total_moon_mass.png', dpi=300)					
		plt.show()



		###### DO THE SAME AS ABOVE, WITH DELTA-AIC.
		colors = cm.get_cmap('viridis')(np.linspace(0,1,5))
		color_idxs = np.linspace(0,1,5)

		fig = plt.figure(figsize=(6,8))
		ax = plt.subplot(111)

		for moonidx, nmoon in enumerate(np.arange(1,6,1)):
			nmoon_idxs = np.where(nmoons == nmoon)[0]
			nmoon_stable_idxs = np.intersect1d(stable_idxs, nmoon_idxs)
			nmoon_stable_idxs = np.intersect1d(nmoon_stable_idxs, good_PTTV_solution_idxs) #### purge all PTTV == 2.0
			nmoon_deltaBICs = deltaBIC_list[nmoon_stable_idxs]
			nmoon_deltaAICs = deltaAIC_list[nmoon_stable_idxs]
			nmoon_total_masses = Msats_over_Mps[nmoon_stable_idxs]

			plt.scatter(nmoon_total_masses, nmoon_deltaAICs, facecolor=colors[moonidx], edgecolor='k', alpha=0.5, s=20, label='N = '+str(nmoon), zorder=0)
		plt.plot(np.logspace(-5,-2,100), np.linspace(-2,-2,100), c='r', alpha=0.5, linestyle='--', linewidth=3, zorder=5)

		plt.xlabel(r'$(\Sigma \, M_S) / M_P$')
		plt.xlim(1e-5,1e-2)
		plt.ylim(-1000, 20)
		plt.xscale('log')
		#plt.yscale('log')
		plt.ylabel(r'$\Delta$ AIC')
		plt.legend(loc=3)
		ax.set_rasterization_zorder(0)
		plt.subplots_adjust(left=0.15, bottom=0.09, right=0.85, top=0.95, wspace=0.2, hspace=0.2)
		if (sim_prefix == '') or (sim_prefix == '_'):
			plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/LAST_RUN/variable_host_deltaAIC_vs_total_moon_mass.pdf', dpi=300)		
			plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/LAST_RUN/variable_host_deltaAIC_vs_total_moon_mass.png', dpi=300)		
		else:
			plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/'+sim_prefix+'/deltaAIC_vs_total_moon_mass.pdf', dpi=300)
			plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/'+sim_prefix+'/deltaAIC_vs_total_moon_mass.png', dpi=300)					
		plt.show()











		#### PLOT DeltaBIC as a function of # OF TRANSITS (proxy is one over period), color coded by nmoons
		colors = cm.get_cmap('viridis')(np.linspace(0,1,5))
		color_idxs = np.linspace(0,1,5)

		fig = plt.figure(figsize=(6,8))
		ax = plt.subplot(111)

		for moonidx, nmoon in enumerate(np.arange(1,6,1)):
			nmoon_idxs = np.where(nmoons == nmoon)[0]
			nmoon_stable_idxs = np.intersect1d(stable_idxs, nmoon_idxs)
			nmoon_stable_idxs = np.intersect1d(nmoon_stable_idxs, good_PTTV_solution_idxs) 
			nmoon_deltaBICs = deltaBIC_list[nmoon_stable_idxs]
			nmoon_deltaAICs = deltaAIC_list[nmoon_stable_idxs]
			nmoon_total_masses = Msats_over_Mps[nmoon_stable_idxs]
			nmoon_periods = np.array(summaryfile['Pplan_days'])[nmoon_stable_idxs]
			nmoon_ntransits = np.floor(3650 / nmoon_periods)

			plt.scatter(nmoon_ntransits, nmoon_deltaBICs, facecolor=colors[moonidx], edgecolor='k', alpha=0.5, s=20, label='N = '+str(nmoon), zorder=0)
		#plt.plot(np.logspace(-5,-2,100), np.linspace(-2,-2,100), c='r', alpha=0.5, linestyle='--', linewidth=3, zorder=5)
		plt.plot(np.linspace(np.nanmin(nmoon_ntransits), np.nanmax(nmoon_ntransits), 100), np.linspace(-2,-2,100), c='r', alpha=0.5, linestyle='--', linewidth=3, zorder=5)

		#plt.xlabel(r'$(\Sigma \, M_S) / M_P$')
		#plt.xlabel('Orbital Period [days]')
		plt.xlabel('# of transits')
		plt.ylim(-1000,20)
		#plt.xscale('log')
		#plt.xlim(1e-5,1e-2)
		#plt.xscale('log')
		#plt.yscale('log')
		plt.ylabel(r'$\Delta$ BIC')
		plt.legend(loc=4)
		ax.set_rasterization_zorder=(0)
		plt.subplots_adjust(left=0.15, bottom=0.09, right=0.85, top=0.95, wspace=0.2, hspace=0.2)		
		if (sim_prefix == '') or (sim_prefix == '_'):
			plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/LAST_RUN/variable_host_deltaBIC_vs_ntransits.pdf', dpi=300)		
			plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/LAST_RUN/variable_host_deltaBIC_vs_ntransits.png', dpi=300)	
		else:
			plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/'+sim_prefix+'/deltaBIC_vs_ntransits.pdf', dpi=300)
			plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/'+sim_prefix+'/deltaBIC_vs_ntransits.png', dpi=300)								
		plt.show()


		#### PLOT DeltaBIC as a function of # OF TRANSITS (proxy is one over period), color coded by nmoons
		colors = cm.get_cmap('viridis')(np.linspace(0,1,5))
		color_idxs = np.linspace(0,1,5)

		fig = plt.figure(figsize=(6,8))
		ax = plt.subplot(111)

		for moonidx, nmoon in enumerate(np.arange(1,6,1)):
			nmoon_idxs = np.where(nmoons == nmoon)[0]
			nmoon_stable_idxs = np.intersect1d(stable_idxs, nmoon_idxs)
			nmoon_stable_idxs = np.intersect1d(nmoon_stable_idxs, good_PTTV_solution_idxs) 
			nmoon_deltaBICs = deltaBIC_list[nmoon_stable_idxs]
			nmoon_deltaAICs = deltaAIC_list[nmoon_stable_idxs]
			nmoon_total_masses = Msats_over_Mps[nmoon_stable_idxs]
			nmoon_periods = np.array(summaryfile['Pplan_days'])[nmoon_stable_idxs]
			nmoon_ntransits = np.floor(3650 / nmoon_periods)

			plt.scatter(nmoon_periods, nmoon_deltaBICs, facecolor=colors[moonidx], edgecolor='k', alpha=0.5, s=20, label='N = '+str(nmoon), zorder=0)
		#plt.plot(np.logspace(-5,-2,100), np.linspace(-2,-2,100), c='r', alpha=0.5, linestyle='--', linewidth=3, zorder=5)
		plt.plot(np.linspace(np.nanmin(nmoon_periods), np.nanmax(nmoon_periods), 100), np.linspace(-2,-2,100), c='r', alpha=0.5, linestyle='--', linewidth=3, zorder=5)

		#plt.xlabel(r'$(\Sigma \, M_S) / M_P$')
		#plt.xlabel('Orbital Period [days]')
		plt.xlabel('Period [days]')
		#plt.ylim(-1000,20)
		#plt.xscale('log')
		#plt.xlim(1e-5,1e-2)
		#plt.xscale('log')
		#plt.yscale('log')
		plt.ylabel(r'$\Delta$ BIC')
		plt.legend(loc=4)
		ax.set_rasterization_zorder(0)
		plt.subplots_adjust(left=0.125, bottom=0.09, right=0.9, top=0.95, wspace=0.2, hspace=0.2)		
		if (sim_prefix == '') or (sim_prefix == '_'):
			plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/LAST_RUN/variable_host_deltaBIC_vs_Pplan.pdf', dpi=300)	
			plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/LAST_RUN/variable_host_deltaBIC_vs_Pplan.png', dpi=300)	
		else:
			plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/'+sim_prefix+'/deltaBIC_vs_Pplan.pdf', dpi=300)
			plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/'+sim_prefix+'/deltaBIC_vs_Pplan.png', dpi=300)						

		plt.show()






		#### PLOT DeltaBIC as a function of TRANSIT DURATIONS (proxy is one over period), color coded by nmoons
		colors = cm.get_cmap('viridis')(np.linspace(0,1,5))
		color_idxs = np.linspace(0,1,5)

		fig = plt.figure(figsize=(6,8))
		ax = plt.subplot(111)

		for moonidx, nmoon in enumerate(np.arange(1,6,1)):
			nmoon_idxs = np.where(nmoons == nmoon)[0]
			nmoon_stable_idxs = np.intersect1d(stable_idxs, nmoon_idxs)
			nmoon_stable_idxs = np.intersect1d(nmoon_stable_idxs, good_PTTV_solution_idxs) 
			nmoon_deltaBICs = deltaBIC_list[nmoon_stable_idxs]
			nmoon_deltaAICs = deltaAIC_list[nmoon_stable_idxs]
			nmoon_total_masses = Msats_over_Mps[nmoon_stable_idxs]
			nmoon_periods = np.array(summaryfile['Pplan_days'])[nmoon_stable_idxs]
			nmoon_Tdurs = sim_Tdurs_hours[nmoon_stable_idxs]
			nmoon_ntransits = np.floor(3650 / nmoon_periods)

			plt.scatter(nmoon_Tdurs, nmoon_deltaBICs, facecolor=colors[moonidx], edgecolor='k', alpha=0.5, s=20, label='N = '+str(nmoon), zorder=0)
		#plt.plot(np.logspace(-5,-2,100), np.linspace(-2,-2,100), c='r', alpha=0.5, linestyle='--', linewidth=3, zorder=5)
		plt.plot(np.linspace(np.nanmin(nmoon_Tdurs), np.nanmax(nmoon_Tdurs), 100), np.linspace(-2,-2,100), c='r', alpha=0.5, linestyle='--', linewidth=3, zorder=5)
		#plt.set_rasterzation_zorder(0)
		#plt.xlabel(r'$(\Sigma \, M_S) / M_P$')
		#plt.xlabel('Orbital Period [days]')
		plt.xlabel('Transit Duration [hours]')
		plt.ylim(-1000,20)
		#plt.xscale('log')
		#plt.xlim(1e-5,1e-2)
		#plt.xscale('log')
		#plt.yscale('log')
		plt.ylabel(r'$\Delta$ BIC')
		plt.legend(loc=4)
		plt.subplots_adjust(left=0.125, bottom=0.09, right=0.9, top=0.95, wspace=0.2, hspace=0.2)			
		if (sim_prefix == '') or (sim_prefix == '_'):
			plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/LAST_RUN/variable_host_deltaBIC_vs_Tdur.pdf', dpi=300)	
			plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/LAST_RUN/variable_host_deltaBIC_vs_Tdur.png', dpi=300)		
		else:
			plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/'+sim_prefix+'/deltaBIC_vs_Tdur.pdf', dpi=300)
			plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/'+sim_prefix+'/deltaBIC_vs_Tdur.png', dpi=300)						
		plt.show()



		#### LOOK AT THE CORRELATION BETWEEN TOTAL MOON MASS AND NTRANSITS / ORBITAL PERIOD.


		#### PLOT DeltaBIC as a function of TRANSIT DURATIONS (proxy is one over period), color coded by nmoons
		colors = cm.get_cmap('viridis')(np.linspace(0,1,5))
		color_idxs = np.linspace(0,1,5)

		fig = plt.figure(figsize=(6,8))
		ax = plt.subplot(111)

		for moonidx, nmoon in enumerate(np.arange(1,6,1)):
			nmoon_idxs = np.where(nmoons == nmoon)[0]
			nmoon_stable_idxs = np.intersect1d(stable_idxs, nmoon_idxs)
			nmoon_stable_idxs = np.intersect1d(nmoon_stable_idxs, good_PTTV_solution_idxs) 
			nmoon_deltaBICs = deltaBIC_list[nmoon_stable_idxs]
			nmoon_deltaAICs = deltaAIC_list[nmoon_stable_idxs]
			nmoon_total_masses = Msats_over_Mps[nmoon_stable_idxs]
			nmoon_periods = np.array(summaryfile['Pplan_days'])[nmoon_stable_idxs]
			nmoon_Tdurs = sim_Tdurs_hours[nmoon_stable_idxs]
			nmoon_ntransits = np.floor(3650 / nmoon_periods)

			plt.scatter(nmoon_total_masses, nmoon_periods, facecolor=colors[moonidx], edgecolor='k', alpha=0.5, s=20, label='N = '+str(nmoon), zorder=0)
		#plt.plot(np.logspace(-5,-2,100), np.linspace(-2,-2,100), c='r', alpha=0.5, linestyle='--', linewidth=3, zorder=5)
		#plt.plot(np.linspace(np.nanmin(nmoon_Tdurs), np.nanmax(nmoon_Tdurs), 100), np.linspace(-2,-2,100), c='r', alpha=0.5, linestyle='--', linewidth=3, zorder=5)
		#plt.set_rasterization_zorder(0)
		plt.xlabel(r'$(\Sigma \, M_S) / M_P$')
		plt.ylabel('Orbital Period [days]')
		plt.ylabel(r'$\Delta$ BIC')
		plt.legend(loc=4)
		plt.subplots_adjust(left=0.125, bottom=0.09, right=0.9, top=0.95, wspace=0.2, hspace=0.2)			
		plt.show()





		##### PLOT THE FRACTION OF DETECTABLE (deltaBIC <= -2) systems, as a function of mass
		stable_detectable_BIC_idxs = np.intersect1d(stable_idxs, np.where(deltaBIC <= -2)[0])
		stable_undetectable_BIC_idxs = np.intersect1d(stable_idxs, np.where(deltaBIC > -2)[0])
		stable_detectable_AIC_idxs = np.intersect1d(stable_idxs, np.where(deltaAIC < 0)[0])
		stable_undetectable_AIC_idxs = np.intersect1d(stable_idxs, np.where(deltaAIC >= 0)[0])		

		if use_BIC_or_AIC.lower() == 'b':
			stable_detectable_idxs = stable_detectable_BIC_idxs
			stable_undetectable_idxs = stable_undetectable_BIC_idxs
		else:
			stable_detectable_idxs = stable_detectable_AIC_idxs
			stable_undetectable_idxs = stable_undetectable_AIC_idxs			


		#### make a histogram of stable and detectable
		stable_detectable_histogram = plt.hist(Msats_over_Mps[stable_detectable_idxs], bins=np.logspace(-5,-2,20), facecolor='DodgerBlue', edgecolor='k', alpha=0.5)
		plt.title('stable and detectable')
		plt.xlabel(r'$(\Sigma \, M_S) / M_P$')
		plt.show()

		stable_histogram = plt.hist(Msats_over_Mps, bins=np.logspace(-5,-2,20), facecolor='DodgerBlue', edgecolor='k', alpha=0.5)
		plt.xlabel(r'$(\Sigma \, M_S) / M_P$')
		plt.title('all stable')
		plt.show()

		#### PLOT FRACTION DETECTABLE
		fraction_detectable = stable_detectable_histogram[0] / stable_histogram[0]
		plt.plot(np.logspace(-5,-2,len(fraction_detectable)), fraction_detectable, color='LightCoral', linewidth=2)
		plt.xlabel(r'$(\Sigma \, M_S) / M_P$')
		plt.ylabel('Fraction Detectable')
		plt.xscale('log')
		plt.show()		

	except KeyboardInterrupt:
		raise KeyboardInterrupt


	except:
		print('could not plot deltaBIC as a function of total moon mass.')






	try:

		#### PLOT THE DELTA-rvals against megno and spock probs
		fig, ax = plt.subplots(2)
		ax[0].scatter(megno_vals[stable_megno_idxs], max_fractional_delta_rvals[stable_megno_idxs], facecolor='DodgerBlue', edgecolor='k', alpha=0.7, s=20)
		ax[0].scatter(megno_vals[unstable_megno_idxs], max_fractional_delta_rvals[unstable_megno_idxs], facecolor='LightCoral', edgecolor='k', alpha=0.7, s=20)
		ax[0].set_xlabel('MEGNO')
		ax[0].set_ylabel(r'max $(\Delta r / \overline{r})$')
		ax[1].scatter(spockprobs[stable_spockprobs], max_fractional_delta_rvals[stable_spockprobs], facecolor='DodgerBlue', edgecolor='k', alpha=0.7, s=20)
		ax[1].scatter(spockprobs[unstable_spockprobs], max_fractional_delta_rvals[unstable_spockprobs], facecolor='LightCoral', edgecolor='k', alpha=0.7, s=20)
		ax[1].set_xlabel(r'SPOCK $P_{\mathrm{stable}}$')
		ax[1].set_ylabel(r'max $(\Delta r / \overline{r})$')
		plt.show()


		#### PLOT SPOCK PROBABILITIES AS A FUNCTION OF MSAT_OVER_MP -- color code by number of moons
		#### break it up by number of moons!
		for moon_number in np.arange(1,6,1):
			nmoons_idxs = np.where(nmoons == moon_number)[0]
			#plt.scatter(Msats_over_Mps[nmoons_stable_idxs], spockprobs[nmoons_stable_idxs], edgecolor='k', alpha=0.5, s=20, label=str(moon_number))
			plt.scatter(Msats_over_Mps[nmoons_idxs], spockprobs[nmoons_idxs], edgecolor='k', alpha=0.5, s=20, label=str(moon_number))
		plt.xlabel(r'$\frac{\Sigma M_S}{M_P} $')
		plt.ylabel(r'$P_{\mathrm{stable}}$')
		plt.xscale('log')
		plt.legend()
		plt.tight_layout()
		plt.show()


		##### DO THE SAME AS ABOVE, BUT FOR MEGNO #
		#### break it up by number of moons!
		for moon_number in np.arange(1,6,1):
			nmoons_idxs = np.where(nmoons == moon_number)[0]
			#plt.scatter(Msats_over_Mps[nmoons_stable_idxs], spockprobs[nmoons_stable_idxs], edgecolor='k', alpha=0.5, s=20, label=str(moon_number))
			plt.scatter(Msats_over_Mps[nmoons_idxs], megno_vals[nmoons_idxs], edgecolor='k', alpha=0.5, s=20, label=str(moon_number))
		plt.xlabel(r'$\frac{\Sigma \, M_S}{M_P} $')
		plt.ylabel('MEGNO')
		plt.xscale('log')
		plt.yscale('log')
		plt.legend()
		plt.tight_layout()
		plt.show() 


		#### plot the inferred periods against the BIC values
		plt.scatter(peak_power_periods_list[stable_idxs], deltaBIC_list[stable_idxs], facecolor='DodgerBlue', edgecolor='k', alpha=0.7, s=20)
		plt.xlabel('TTV period [epochs]')
		plt.ylabel(r'$\Delta \mathrm{BIC}$')
		plt.yscale('log')
		plt.show()


		##### DO THE SAME WITH THE TTV RMS AMPLITUDES
		plt.scatter(np.array(summaryfile['TTV_rmsamp_sec'])[stable_idxs]/60, deltaBIC_list[stable_idxs], facecolor='DodgerBlue', edgecolor='k', alpha=0.7, s=20)
		plt.xlabel('TTV r.m.s. [minutes]')
		plt.ylabel(r'$\Delta \mathrm{BIC}$')
		plt.show()	


		##### TTV RMS vs Planet Period
		Pplans = np.array(summaryfile['Pplan_days']).astype(float)
		plt.scatter(Pplans[stable_idxs], np.array(summaryfile['TTV_rmsamp_sec'])[stable_idxs]/60, facecolor='DodgerBlue', edgecolor='k', alpha=0.7, s=20)
		plt.xlabel(r'$P_P$ [days]')
		plt.ylabel('TTV r.m.s. [minutes]')
		plt.show()	




		#### break it up by number of moons!
		for moon_number in np.arange(1,6,1):
			nmoons_idxs = np.where(nmoons == moon_number)[0]
			nmoons_stable_idxs = np.intersect1d(nmoons_idxs, stable_idxs)
			plt.scatter(peak_power_periods_list[nmoons_stable_idxs], deltaBIC_list[nmoons_stable_idxs], edgecolor='k', alpha=0.5, s=20, label=str(moon_number))
		plt.plot(np.linspace(2,500,1000), np.linspace(0,0,1000), c='k', linestyle='--')
		plt.xlabel('TTV period [epochs]')
		plt.ylabel(r'$\Delta \mathrm{BIC}$')
		plt.xscale('log')
		plt.legend()
		plt.tight_layout()
		plt.show()

	except KeyboardInterrupt:
		raise KeyboardInterrupt

	except:
		print('Could not produce other population plots.')



	#### fit exponential curves to these histogram points
	def TTV_curve(xvals, amplitude, beta):
		#### returns an function of the form amplitude * np.exp(-xvals / beta)
		return amplitude * np.exp(-xvals / beta)


	fig, ax = plt.subplots(5, sharex=True, figsize=(6,8))
	colors = cm.get_cmap('viridis')(np.linspace(0,1,5))
	color_idxs = np.linspace(0,1,5)
	histdict = {}
	for moon_number in np.arange(1,6,1):
		nmoon_idxs = np.where(nmoons == moon_number)[0]
		good_BIC_idxs = np.where(deltaBIC_list < -2)[0] #### positive evidence for a moon
		good_AIC_idxs = np.where(deltaAIC_list < 0)[0]
		nmoons_stable_idxs = np.intersect1d(nmoon_idxs, stable_idxs)
		nmoons_stable_good_BIC_idxs = np.intersect1d(nmoons_stable_idxs, good_BIC_idxs)
		nmoons_stable_good_AIC_idxs = np.intersect1d(nmoon_stable_idxs, good_AIC_idxs)
		if len(cv_robust_idxs) > 0:
			if use_BIC_or_AIC.lower() == 'b':
				nmoons_final_sample_idxs = np.intersect1d(nmoons_stable_good_BIC_idxs, cv_robust_idxs)
			elif use_BIC_or_AIC.lower() == 'a':
				nmoons_final_sample_idxs = np.intersect1d(nmoons_stable_good_AIC_idxs, cv_robust_idxs)
		else:
			if use_BIC_or_AIC.lower() == 'b':
				nmoons_final_sample_idxs = nmoons_stable_good_BIC_idxs
			else:
				nmoons_final_sample_idxs = nmoons_stable_good_AIC_idxs				
		TTV_period_bins = np.arange(2,20,1)


		print('sim_prefix: ', sim_prefix)
		print('moon number = ', moon_number)
		print('# of systems = ', len(nmoon_idxs))
		print('# of these that are stable = ', len(nmoons_stable_idxs))
		print('percent stable = ', len(nmoons_stable_idxs) / len(nmoon_idxs))
		print('# stable with good BIC = ', len(nmoons_stable_good_BIC_idxs))
		print('# stable with good AIC = ', len(nmoons_stable_good_AIC_idxs))		
		print('# stable, good (A)(B)IC, robust cross-validation = ', len(nmoons_final_sample_idxs))
		print(" ")
		
		histdict['hist'+str(moon_number)] = ax[moon_number-1].hist(peak_power_periods_list[nmoons_final_sample_idxs], bins=TTV_period_bins, facecolor=colors[moon_number-1], edgecolor='k')
		ax[moon_number-1].set_ylabel(r'$N = $'+str(moon_number))

		nperbin = histdict['hist'+str(moon_number)][0]

		if np.all(nperbin == 0):
			continue

		expo_curve_popt, expo_curve_pcov = curve_fit(TTV_curve, TTV_period_bins[:-1]+0.5, nperbin, bounds=([0.5*np.nanmax(nperbin), 1e-1], [4*np.nanmax(nperbin), 10]))
		#except:
		#	#### unbounded
		#	expo_curve_popt, expo_curve_pcov = curve_fit(TTV_curve, TTV_period_bins[:-1]+0.5, nperbin, bounds=([0, 1e-1], [4*np.nanmax(nperbin), 10]))

		TTV_bins_smooth = np.linspace(TTV_period_bins[0], TTV_period_bins[-1], 1000)
		TTV_hist_vals = TTV_curve(TTV_bins_smooth, *expo_curve_popt)
		ax[moon_number-1].plot(TTV_bins_smooth, TTV_hist_vals, c='k', linestyle='--')	
		ax[moon_number-1].set_ylim(0,1.1*np.nanmax(nperbin))
		ax[moon_number-1].text(14, 0.95*np.nanmax(nperbin), r'$\propto \mathrm{exp}(-$'+str(round(1/expo_curve_popt[1],2))+r'$\, P_{\mathrm{TTV}})$')

		print('N = '+str(moon_number)+'; y = '+str(round(expo_curve_popt[0],2))+' * e^(-'+str(round(1/expo_curve_popt[1],2))+'x)')

	ax[4].set_xlabel('TTV period [epochs]')
	#plt.tight_layout()
	plt.subplots_adjust(left=0.135, bottom=0.058, right=0.888, top=0.985, wspace=0.05, hspace=0.05)		
	if (sim_prefix == '') or (sim_prefix == '_'):
		plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/LAST_RUN/variable_host_exomoon_corridor_by_architecture.pdf', dpi=300)	
		plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/LAST_RUN/variable_host_exomoon_corridor_by_architecture.png', dpi=300)		
	else:
		plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/'+sim_prefix+'/exomoon_corridor_by_architecture.pdf', dpi=300)
		plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/'+sim_prefix+'/exomoon_corridor_by_architecture.png', dpi=300)			

	plt.show()




	######## DO THE EXACT SAME THING AS ABOVE, BUT WITH A GEOMETRIC BIAS APPLIED!!!!!!
	fig, ax = plt.subplots(5, sharex=True, figsize=(6,8))
	colors = cm.get_cmap('viridis')(np.linspace(0,1,5))
	color_idxs = np.linspace(0,1,5)
	histdict = {}
	for moon_number in np.arange(1,6,1):
		nmoon_idxs = np.where(nmoons == moon_number)[0]
		good_BIC_idxs = np.where(deltaBIC_list < -2)[0] #### positive evidence for a moon
		good_AIC_idxs = np.where(deltaAIC_list < 0)[0] #### positive evidence for a moon		
		nmoons_stable_idxs = np.intersect1d(nmoon_idxs, stable_idxs)
		if use_BIC_or_AIC.lower() == 'b':
			nmoons_stable_good_BIC_idxs = np.intersect1d(nmoons_stable_idxs, good_BIC_idxs)
		else:
			nmoons_stable_good_AIC_idxs = np.intersect1d(nmoons_stable_idxs, good_AIC_idxs)			


		if len(cv_robust_idxs) > 0:
			if use_BIC_or_AIC.lower() == 'b':
				nmoons_final_sample_idxs = np.intersect1d(nmoons_stable_good_BIC_idxs, cv_robust_idxs)
			else:
				nmoons_final_sample_idxs = np.intersect1d(nmoons_stable_good_AIC_idxs, cv_robust_idxs)				
		else:
			if use_BIC_or_AIC.lower() == 'b':
				nmoons_final_sample_idxs = nmoons_stable_good_BIC_idxs
			else:
				nmoons_final_sample_idxs = nmoons_stable_good_AIC_idxs				

		###### ADD GEOMETRIC BIAS
		nmoon_transit_probs = scaled_transit_probabilities[nmoons_final_sample_idxs] #### we've just scaled these probabilities so that the closest in planet has transit probability of one.
		##### compare against random draws
		random_draws = np.random.random(size=len(nmoon_transit_probs))
		transit_probs_minus_random_draws = nmoon_transit_probs - random_draws
		##### include instances where the above values are positive -- you beat the odds.
		actual_transit_idxs = np.where(transit_probs_minus_random_draws > 0)[0]
		nmoons_final_sample_idxs = nmoons_final_sample_idxs[actual_transit_idxs] #### 

		TTV_period_bins = np.arange(2,20,1)

		print('sim_prefix: ', sim_prefix)
		print("GEOMETRIC BIAS APPLIED!")
		print('moon number = ', moon_number)
		print('# of systems = ', len(nmoon_idxs))
		print('# of these that are stable = ', len(nmoons_stable_idxs))
		print('percent stable = ', len(nmoons_stable_idxs) / len(nmoon_idxs))
		print('# stable with good BIC = ', len(nmoons_stable_good_BIC_idxs))
		print('# stable with good AIC = ', len(nmoons_stable_good_AIC_idxs))		
		print('# stable, good BIC, robust cross-validation = ', len(nmoons_final_sample_idxs))
		print(" ")
		
		histdict['hist'+str(moon_number)] = ax[moon_number-1].hist(peak_power_periods_list[nmoons_final_sample_idxs], bins=TTV_period_bins, facecolor=colors[moon_number-1], edgecolor='k')
		ax[moon_number-1].set_ylabel(r'$N = $'+str(moon_number))

		nperbin = histdict['hist'+str(moon_number)][0]

		if np.all(nperbin == 0):
			continue

		expo_curve_popt, expo_curve_pcov = curve_fit(TTV_curve, TTV_period_bins[:-1]+0.5, nperbin, bounds=([0.5*np.nanmax(nperbin), 1e-1], [4*np.nanmax(nperbin), 10]))
		#except:
		#	#### unbounded
		#	expo_curve_popt, expo_curve_pcov = curve_fit(TTV_curve, TTV_period_bins[:-1]+0.5, nperbin, bounds=([0, 1e-1], [4*np.nanmax(nperbin), 10]))

		TTV_bins_smooth = np.linspace(TTV_period_bins[0], TTV_period_bins[-1], 1000)
		TTV_hist_vals = TTV_curve(TTV_bins_smooth, *expo_curve_popt)
		ax[moon_number-1].plot(TTV_bins_smooth, TTV_hist_vals, c='k', linestyle='--')	
		ax[moon_number-1].set_ylim(0,1.1*np.nanmax(nperbin))
		ax[moon_number-1].text(14, 0.95*np.nanmax(nperbin), r'$\propto \mathrm{exp}(-$'+str(round(1/expo_curve_popt[1],2))+r'$\, P_{\mathrm{TTV}})$')

		print('N = '+str(moon_number)+'; y = '+str(round(expo_curve_popt[0],2))+' * e^(-'+str(round(1/expo_curve_popt[1],2))+'x)')

	ax[4].set_xlabel('TTV period [epochs]')
	#plt.tight_layout()
	plt.subplots_adjust(left=0.135, bottom=0.058, right=0.888, top=0.985, wspace=0.05, hspace=0.05)	
	if (sim_prefix == '') or (sim_prefix == '_'):
		plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/LAST_RUN/variable_host_exomoon_corridor_by_architecture_with_geometric_bias.pdf', dpi=300)		
		plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/LAST_RUN/variable_host_exomoon_corridor_by_architecture_with_geometric_bias.png', dpi=300)	
	else:
		plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/'+sim_prefix+'/exomoon_corridor_by_architecture_with_geometric_bias.pdf', dpi=300)
		plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/'+sim_prefix+'/exomoon_corridor_by_architecture_with_geometric_bias.png', dpi=300)				

	plt.show()















	
	#### GENERATE THE SAME PLOT, BUT WITH A GAUSSIAN KERNEL DENSITY ESTIMATOR (COMPARE TO RESULTS FROM "real_planet_TTV_analyzer.py")
	
	good_BIC_stable_idxs = np.intersect1d(good_BIC_idxs, stable_idxs)
	good_AIC_stable_idxs = np.intersect1d(good_AIC_idxs, stable_idxs)
	if use_BIC_or_AIC.lower() == 'b':	
		P_plans = np.array(summaryfile['Pplan_days'])[good_BIC_stable_idxs]
		P_TTVs = peak_power_periods_list[good_BIC_stable_idxs]
	else:
		P_plans = np.array(summaryfile['Pplan_days'])[good_AIC_stable_idxs]
		P_TTVs = peak_power_periods_list[good_AIC_stable_idxs]		
	kdestack = np.vstack((P_plans, P_TTVs))


	"""
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
	xbins = np.logspace(np.log10(10), np.log10(1500), 20) #### planet periods
	ybins = np.logspace(np.log10(2), np.log10(100), 20) #### P_TTVs
	TTV_Pplan_hist2d = np.histogram2d(P_plans, P_TTVs, bins=[xbins, ybins])
	plt.imshow(TTV_Pplan_hist2d[0].T, origin='lower', cmap=cm.coolwarm)
	plt.xticks(ticks=np.arange(0,len(xbins),5), labels=np.around(np.log10(xbins[::5]),2))
	plt.yticks(ticks=np.arange(0,len(ybins),5), labels=np.around(np.log10(ybins[::5]), 2))
	plt.xlabel(r'$\log_{10} \, P_{\mathrm{P}}$ [days]')
	plt.ylabel(r'$\log_{10} \, P_{\mathrm{TTV}}$ [epochs]')
	plt.title('Numpy 2D histogram')
	plt.tight_layout()
	plt.show()

	#### COMPARE TO NATIVE MATPLOTLIB HISTOGRAM
	#### THIS IS MUCH BETTER -- you get the tick labels for free... could even do a scatter over top
	plt.figure(figsize=(6,6))
	plt.hist2d(P_plans, P_TTVs, bins=[xbins, ybins], cmap='coolwarm')
	plt.scatter(P_plans, P_TTVs, facecolor='w', edgecolor='k', s=5, alpha=0.3)
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel(r'$P_{\mathrm{P}}$ [days]')
	plt.ylabel(r'$P_{\mathrm{TTV}}$ [epochs]')
	#plt.title('Matplotlib 2D histogram')
	plt.show()
	"""




	#np.save('/data/tethys/Documents/Projects/NMoon_TTVs/simulated_PTTV10-1500_Pplan2-100_20x20_heatmap.npy', TTV_Pplan_hist2d)











	#### SCATTER PLOTS
	"""
	plt.scatter(megno_vals, deltaBIC_list, facecolor='DodgerBlue', edgecolor='k', alpha=0.7)
	plt.plot(np.linspace(np.nanmin(megno_vals), np.nanmax(megno_vals), 100), np.linspace(0,0,100), c='k', linestyle='--')
	plt.xlabel('MEGNO')
	plt.ylabel(r'$\Delta \mathrm{BIC}$')
	plt.xscale('log')
	plt.show()
	"""

	"""
	plt.scatter(spockprobs*100, deltaBIC_list, facecolor='DodgerBlue', edgecolor='k', alpha=0.7)
	plt.xlabel('survival probability [%]')
	plt.ylabel(r'$\Delta \mathrm{BIC}$')
	plt.show()
	"""


	###### HISTOGRAMS
	######### THE STABLE CASE!!!!! -- COMPARING DISTRIBUTIONS FOR MEGNO AND SPOCK PROBABILITIES
	"""
	fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(6,10))
	n1,b1,e1 = ax1.hist(deltaBIC_list[stable_megno_idxs], bins=np.arange(-10,11,1), facecolor='NavajoWhite', edgecolor='k', alpha=1, zorder=1)
	ax1.plot(np.linspace(np.nanmedian(deltaBIC_list[stable_megno_idxs]),np.nanmedian(deltaBIC_list[stable_megno_idxs]),100), np.linspace(0,1.1*np.nanmax(n1),100), c='k', linestyle='--')
	ax1.axvspan(-11, 0, alpha=0.2, color='green', zorder=0)
	ax1.axvspan(0, 11, alpha=0.2, color='red', zorder=0)
	ax1.set_ylabel(str(megno_min)+r'$\leq \mathrm{MEGNO} \leq $'+str(megno_max))
	ax1.set_xlim(-10,10)
	ax1.set_ylim(0,1.1*np.nanmax(n1))
	n2,b2,e2 = ax2.hist(deltaBIC_list[stable_spockprobs], bins=np.arange(-10,11,1), facecolor='NavajoWhite', edgecolor='k', alpha=1, zorder=1)
	ax2.plot(np.linspace(np.nanmedian(deltaBIC_list[stable_spockprobs]),np.nanmedian(deltaBIC_list[stable_spockprobs]),100), np.linspace(0,1.1*np.nanmax(n2),100), c='k', linestyle='--')
	ax2.axvspan(-11, 0, alpha=0.2, color='green', zorder=0)
	ax2.axvspan(0, 11, alpha=0.2, color='red', zorder=0)
	ax2.set_ylabel(r'SPOCK $P_{\mathrm{stable}} \geq$'+str(spock_min))
	ax2.set_xlim(-10,10)
	ax2.set_ylim(0,1.1*np.nanmax(n2))
	
	ax2.set_xlabel(r'$\Delta \mathrm{BIC}$')
	ax1.set_title('stable systems')
	plt.tight_layout()
	plt.show()
	"""


	###### HISTOGRAMS
	######### THE STABLE CASE!!!!! (COMBINED MEGNO / SPOCK STABLE_IDXs)
	"""
	fig, ax1 = plt.subplots(figsize=(6,6))
	n1,b1,e1 = ax1.hist(deltaBIC_list[stable_idxs], bins=np.arange(-10,11,1), facecolor='NavajoWhite', edgecolor='k', alpha=1, zorder=1)
	ax1.plot(np.linspace(np.nanmedian(deltaBIC_list[stable_idxs]),np.nanmedian(deltaBIC_list[stable_idxs]),100), np.linspace(0,1.1*np.nanmax(n1),100), c='k', linestyle='--')
	all_below_neg2 = np.where(deltaBIC_list[stable_idxs] <= -2)[0]
	pct_below_neg2 = len(all_below_neg2) / len(stable_idxs)

	ax1.axvspan(-11, 0, alpha=0.2, color='green', zorder=0)
	ax1.axvspan(0, 11, alpha=0.2, color='red', zorder=0)
	#ax1.set_ylabel(r'$1.97\leq \mathrm{MEGNO} \leq 2.18$')
	ax1.set_xlim(-10,10)
	ax1.set_ylim(0,1.1*np.nanmax(n1))
	ax1.text(-9, np.nanmax(n1), str(round(pct_below_neg2*100,2))+r'% with $\Delta \mathrm{BIC} \leq -2$')

	ax1.set_xlabel(r'$\Delta \mathrm{BIC}$')
	ax1.set_title('stable systems')
	plt.tight_layout()
	plt.show()
	"""




	###### HISTOGRAMS --- COMPARING *UNSTABLE* MEGNO AND SPOCK DISTRIBUTIONS.
	"""
	fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(6,10))
	n1,b1,e1 = ax1.hist(deltaBIC_list[unstable_megno_idxs], bins=np.arange(-10,11,1), facecolor='NavajoWhite', edgecolor='k', alpha=1, zorder=1)
	ax1.plot(np.linspace(np.nanmedian(deltaBIC_list[unstable_megno_idxs]),np.nanmedian(deltaBIC_list[unstable_megno_idxs]),100), np.linspace(0,1.1*np.nanmax(n1),100), c='k', linestyle='--')
	ax1.axvspan(-11, 0, alpha=0.2, color='green', zorder=0)
	ax1.axvspan(0, 11, alpha=0.2, color='red', zorder=0)
	ax1.set_ylabel(r'$\mathrm{MEGNO} < '+str(megno_min)+' \, \mathrm{;} > $'+str(megno_max))
	ax1.set_xlim(-10,10)
	ax1.set_ylim(0,1.1*np.nanmax(n1))
	n2,b2,e2 = ax2.hist(deltaBIC_list[unstable_spockprobs], bins=np.arange(-10,11,1), facecolor='NavajoWhite', edgecolor='k', alpha=1, zorder=1)
	ax2.plot(np.linspace(np.nanmedian(deltaBIC_list[unstable_spockprobs]),np.nanmedian(deltaBIC_list[unstable_spockprobs]),100), np.linspace(0,1.1*np.nanmax(n2),100), c='k', linestyle='--')
	ax2.axvspan(-11, 0, alpha=0.2, color='green', zorder=0)
	ax2.axvspan(0, 11, alpha=0.2, color='red', zorder=0)
	ax2.set_ylabel(r'SPOCK $P_{\mathrm{stable}} < $'+str(spock_min))
	ax2.set_xlim(-10,10)
	ax2.set_ylim(0,1.1*np.nanmax(n2))
	
	ax2.set_xlabel(r'$\Delta \mathrm{BIC}$')
	ax1.set_title('unstable systems')
	plt.tight_layout()
	plt.show()
	"""


	TTV_amplitudes = np.array(TTV_amplitudes)
	TTV_amplitudes_minutes = TTV_amplitudes / 60
	#### SAVE THIS STUFF
	np.save(projectdir+'/'+sim_prefix_+'sim_deltaBIC_list.npy', deltaBIC_list[good_BIC_stable_idxs])
	np.save(projectdir+'/'+sim_prefix_+'sim_deltaAIC_list.npy', deltaAIC_list[good_AIC_stable_idxs])	
	np.save(projectdir+'/'+sim_prefix_+'sim_PTTVs.npy', P_TTVs) #### these have already been culled as good_BIC_stable_idxs
	np.save(projectdir+'/'+sim_prefix_+'sim_Pplans.npy', P_plans)
	if use_BIC_or_AIC.lower() == 'b':
		np.save(projectdir+'/'+sim_prefix_+'sim_TTV_amplitudes.npy', TTV_amplitudes[good_BIC_stable_idxs]) #### these need culling as such.
	else:
		np.save(projectdir+'/'+sim_prefix_+'sim_TTV_amplitudes.npy', TTV_amplitudes[good_AIC_stable_idxs]) #### these need culling as such.		
	#P_plans = np.array(summaryfile['Pplan_days'])[good_BIC_stable_idxs]
	#P_TTVs = peak_power_periods_list[good_BIC_stable_idxs]


	if use_BIC_or_AIC.lower() == 'b':
		detectable_idxs = np.where(deltaBIC_list <= -2)[0]
	else:
		detectable_idxs = np.where(deltaAIC_list <= 0)[0]		
	P_TTVs = peak_power_periods_list
	#### final statistics:
	for nmoon in np.arange(1,6,1):
		"""
		print("Moon #: ", nmoon)
		nmoon_idxs = np.where(nmoons == nmoon)[0] ### len == number of models with this architecture
		nmoon_deltaBICs = deltaBIC_list[nmoon_idxs]  #### len == number of models with this architecture
		nmoon_stable_bool = stable_bool[nmoon_idxs] 
		nmoon_robust_bool = sim_cv_results_robust_bool[nmoon_idxs]
		nmoon_stable_idxs = np.where(nmoon_deltaBICs <= -2)[0]
		nmoon_nstable = len(nmoon_stable_idxs)
		nmoon_detectable_idxs = np.where(nmoon_deltaBICs <= -2)[0]
		nmoon_ndetectable = len(nmoon_detectable_idxs)
		nmoon_robust_idxs = np.where(nmoon_robust_bool)[0]
		nmoon_nrobust = len(nmoon_robust_idxs)
		nmoon_stable_detectable_idxs = np.intersect1d(nmoon_stable_idxs, nmoon_detectable_idxs)
		nmoon_stable_detectable_robust_idxs = np.intersect1d(nmoon_stable_detectable_idxs, nmoon_robust_idxs) 


		print('# of moons: ', len(nmoon_idxs))
		print("percent stable: ", len(nmoon_stable_idxs) / len(nmoon_idxs))
		print("percent stable and detectable: ", len(nmoon_stable_detectable_idxs) / len(nmoon_idxs))
		print("percent stable, detectable, robust: ", len(nmoon_stable_detectable_robust_idxs) / len(nmoon_idxs))
		print('median P_TTV: ', np.nanmedian(peak_power_periods_list[nmoon_stable_detectable_robust_idxs]))
		#print("median A_TTV: ", np.nanmedian(TTV_amplitudes[nmoon_stable_detectable_idxs]))
		print('median M_S / M_P: ', np.nanmedian(np.array(summaryfile['Mmoons_over_Mplan'])[nmoon_stable_detectable_robust_idxs]))
		print(' ')
		print(' ')
		"""


		nmoon_idxs = np.where(nmoons == nmoon)[0]
		
		nmoon_stable_idxs = np.intersect1d(original_stable_idxs, nmoon_idxs)
		nmoon_deltaBICs = deltaBIC_list[nmoon_stable_idxs]
		nmoon_good_deltaBIC_idxs = np.intersect1d(np.where(nmoons == nmoon)[0], np.where(deltaBIC_list <= -2)[0])
		nmoon_deltaAICs = deltaAIC_list[nmoon_stable_idxs]
		nmoon_good_deltaAIC_idxs = np.intersect1d(np.where(nmoons == nmoon)[0], np.where(deltaAIC_list < 0)[0])		
		nmoon_robust_idxs = np.intersect1d(np.where(nmoons == nmoon)[0], np.where(sim_cv_results_robust_bool == True)[0])



		nmoon_good_deltaBIC_and_stable_idxs = np.intersect1d(nmoon_stable_idxs, nmoon_good_deltaBIC_idxs)
		nmoon_good_deltaBIC_stable_and_robust_idxs = np.intersect1d(nmoon_good_deltaBIC_and_stable_idxs, nmoon_robust_idxs)
		nmoon_good_deltaAIC_and_stable_idxs = np.intersect1d(nmoon_stable_idxs, nmoon_good_deltaAIC_idxs)
		nmoon_good_deltaAIC_stable_and_robust_idxs = np.intersect1d(nmoon_good_deltaAIC_and_stable_idxs, nmoon_robust_idxs)

		if use_BIC_or_AIC.lower() == 'b':
			nmoon_final_idxs = nmoon_good_deltaBIC_stable_and_robust_idxs 
		else:
			nmoon_final_idxs = nmoon_good_deltaAIC_stable_and_robust_idxs 			
		
		nmoon_median_PTTV = np.nanmedian(P_TTVs[nmoon_final_idxs])
		nmoon_PTTV_159 = np.nanpercentile(P_TTVs[nmoon_final_idxs], 15.9)
		nmoon_PTTV_841 = np.nanpercentile(P_TTVs[nmoon_final_idxs], 84.1)
		nmoon_PTTV_pm = (nmoon_PTTV_841 - nmoon_median_PTTV, nmoon_median_PTTV - nmoon_PTTV_159)
		
		nmoon_median_ATTV = np.nanmedian(TTV_amplitudes_minutes[nmoon_final_idxs])
		nmoon_ATTV_159 = np.nanpercentile(TTV_amplitudes_minutes[nmoon_final_idxs], 15.9)
		nmoon_ATTV_841 = np.nanpercentile(TTV_amplitudes_minutes[nmoon_final_idxs], 84.1)
		nmoon_ATTV_pm = (nmoon_ATTV_841 - nmoon_median_ATTV, nmoon_median_ATTV - nmoon_ATTV_159)
		
		nmoon_median_MsMp = np.nanmedian(Msats_over_Mps[nmoon_final_idxs])
		nmoon_MsMp_159 = np.nanpercentile(Msats_over_Mps[nmoon_final_idxs], 15.9)
		nmoon_MsMp_841 = np.nanpercentile(Msats_over_Mps[nmoon_final_idxs], 84.1)
		nmoon_MsMp_pm = (nmoon_MsMp_841 - nmoon_median_MsMp, nmoon_median_MsMp - nmoon_MsMp_159)



		print('sim_prefix: ', sim_prefix)
		print('nmoon = ', nmoon)
		print('total available: ', len(nmoon_idxs))
		print('number stable: ', len(nmoon_stable_idxs))
		print('pct stable: ', (len(nmoon_stable_idxs) / len(nmoon_idxs))*100)
		print('number w/ good deltaBIC: ', len(np.where(nmoon_deltaBICs <= -2)[0]))
		print('pct w/ good deltaBIC: ', (len(np.where(nmoon_deltaBICs <= -2)[0]) / len(nmoon_idxs))*100)
		print('number stable AND good deltaBIC: ', len(nmoon_good_deltaBIC_and_stable_idxs))
		print('pct stable AND good deltaBIC: ', (len(nmoon_good_deltaBIC_and_stable_idxs) / len(nmoon_idxs))*100)
		
		print('number w/ good deltaAIC: ', len(np.where(nmoon_deltaAICs < 0 )[0]))
		print('pct w/ good deltaAIC: ', (len(np.where(nmoon_deltaAICs < 0)[0]) / len(nmoon_idxs))*100)
		print('number stable AND good deltaAIC: ', len(nmoon_good_deltaAIC_and_stable_idxs))
		print('pct stable AND good deltaAIC: ', (len(nmoon_good_deltaAIC_and_stable_idxs) / len(nmoon_idxs))*100)

		print('number robust: ', len(nmoon_robust_idxs))

		print('number, stable, detectable, robust (BIC) = ', len(nmoon_good_deltaBIC_stable_and_robust_idxs))
		print('pct stable, detectable, robust (BIC) = ', (len(nmoon_good_deltaBIC_stable_and_robust_idxs) / len(nmoon_idxs))*100)

		print('number, stable, detectable, robust (AIC) = ', len(nmoon_good_deltaAIC_stable_and_robust_idxs))
		print('pct stable, detectable, robust (AIC) = ', (len(nmoon_good_deltaAIC_stable_and_robust_idxs) / len(nmoon_idxs))*100)	
			
		print("PTTV = $"+str(round(nmoon_median_PTTV,2))+'\,^{+'+str(round(nmoon_PTTV_pm[0],2))+'}_{-'+str(round(nmoon_PTTV_pm[1],2))+'}$')
		print("ATTV = $"+str(round(nmoon_median_ATTV,2))+'\,^{+'+str(round(nmoon_ATTV_pm[0],2))+'}_{-'+str(round(nmoon_ATTV_pm[1],2))+'}$')
		print("MsMp = $"+str(round(nmoon_median_MsMp,6))+'\,^{+'+str(round(nmoon_MsMp_pm[0],6))+'}_{-'+str(round(nmoon_MsMp_pm[1],6))+'}$')		
		print(' ')



except KeyboardInterrupt:
	raise KeyboardInterrupt


except:
	traceback.print_exc()
	raise Exception('something happened.')

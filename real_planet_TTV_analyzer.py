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
from astropy.constants import M_sun, M_earth
from sklearn.cluster import DBSCAN
from scipy.stats import skew, kurtosis

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

plt.rcParams["font.family"] = 'serif'

sim_prefix = input('What is the sim_prefix? ')

####### FILE DIRECTORY INFORMATION
if socket.gethostname() == 'tethys.asiaa.sinica.edu.tw':
	#projectdir = '/data/tethys/Documents/Projects/NMoon_TTVs'
	projectdir = '/run/media/amteachey/Auddy_Akiti/Teachey/Nmoon_TTVs'
elif socket.gethostname() == 'Alexs-MacBook-Pro.local':
	projectdir = '/Users/hal9000/Documents/Projects/Nmoon_TTVsim'
else:
	projectdir = input('Please input the project directory: ')

if sim_prefix == '':
	positionsdir = projectdir+'/sim_positions'
	ttvfiledir = projectdir+'/sim_TTVs'
	LSdir = projectdir+'/sim_periodograms'
	modeldictdir = projectdir+'/sim_model_settings'
	plotdir = projectdir+'/sim_plots'

else:
	positionsdir = projectdir+'/'+sim_prefix+'_sim_positions'
	ttvfiledir = projectdir+'/'+sim_prefix+'_sim_TTVs'
	LSdir = projectdir+'/'+sim_prefix+'_sim_periodograms'
	modeldictdir = projectdir+'/'+sim_prefix+'_sim_model_settings'
	plotdir = projectdir+'/'+sim_prefix+'_sim_plots'
###################################



########## FUNCTION DEFINTIONS ####################
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
	if np.isfinite(mstar) and (mstar != 0) and type(mplan) != None:
		q = mplan / mstar
	else:
		q = np.nan 
	first_term = 9 / (q**(1/3))
	second_term = A_TTV / Pplan 
	minimum_fRHill = first_term * second_term

	return minimum_fRHill, q, mplan, mstar, A_TTV, Pplan, first_term, second_term


def n_choose_k(n,k):
	numerator = factorial(n)
	denominator = factorial(k) * factorial(n - k)
	return numerator / denominator

def chisquare(data, model, errors):
	return np.nansum(((data - model)**2) / errors**2)


########### END FUNCTION DEFINTIONS ###############################







try:


	show_plots = input('Do you want to show plots (for debugging)? y/n: ')


	cross_validate_LSPs = input("Do you want to run Lomb-Scargle cross-validation (removing points and recomputing)? ")
	if cross_validate_LSPs == 'y':
		cv_frac_to_leave_out = 0.05 #### five percent
		cv_ntrials = int(1/cv_frac_to_leave_out) ### 20 trials
	elif cross_validate_LSPs == 'n':
		### just do a single trial
		cv_frac_to_leave_out = 0
		cv_ntrials = 1

	exclude_short_periods = input('Do you want to exclude short period planets? (recommend y): ')


	################ LOAD TTVs ##################################

	Holczer_OCfile = pandas.read_csv('/data/tethys/Documents/Software/MoonPy/Table3_O-C.csv')
	Holczer_KOIs = np.array(Holczer_OCfile['KOI']).astype(str)
	Holczer_epochs = np.array(Holczer_OCfile['n']).astype(int)
	Holczer_OCmin = np.array(Holczer_OCfile['O-C_min']).astype(str)
	Holczer_OCmin_err = np.array(Holczer_OCfile['O-C_err']).astype(str)

	### PURGE THOSE GODDAMN SPECIAL CHARACTERS OUT OF Holczer_OCmin and Holczer_OCmin_err
	Holczer_OCmin_clean, Holczer_OCmin_err_clean = [], []

	for OC, OCerr in zip(Holczer_OCmin, Holczer_OCmin_err):
		### THIS NASTY NESTED FOR LOOP IS BROUGHT TO YOU BY USING NON-NUMERIC CHARACTERS IN A NUMERIC COLUMN INSTEAD OF USING FLAGS.
		OCclean = ''
		for val in OC:
			if val in ['-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
				OCclean = OCclean+val
			else:
				pass
		Holczer_OCmin_clean.append(float(OCclean))

		OCerrclean = ''	
		for val in OCerr:
			if val in ['-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
				OCerrclean = OCerrclean+val
			else:
				pass
		Holczer_OCmin_err_clean.append(float(OCerrclean))

	Holczer_OCmin = np.array(Holczer_OCmin_clean)
	Holczer_OCmin_err = np.array(Holczer_OCmin_err_clean)

	Holczer_unique_KOIs = np.unique(Holczer_KOIs)


	#### LOAD TKS (GOSE PROJECT) TTVs (Teachey, Kipping & Schmitt) ################################
	TKS_OCfile = pandas.read_csv('/data/tethys/Documents/Central_Data/GOSE_TTV_summaryfile.csv')
	TKS_KOIs = np.array(TKS_OCfile['KOI']).astype(str)
	TKS_KOI_nums = []
	for TKS_KOI in TKS_KOIs:
		TKS_KOI_nums.append(str(TKS_KOI[4:]))
	TKS_KOIs = np.array(TKS_KOI_nums).astype(str)
	TKS_epochs = np.array(TKS_OCfile['epoch']).astype(int)
	TKS_OCmin = np.array(TKS_OCfile['OC_min']).astype(str)
	TKS_OCmin_err = np.array(TKS_OCfile['OCmin_err']).astype(str)
	TKS_unique_KOIs = np.unique(TKS_KOIs)

	use_Holczer_or_gose = input("Do you want to use 'h'olczer TTVs or 'g'ose (TKS) TTVs? ")
	if use_Holczer_or_gose == 'h':
		OCfile = Holczer_OCfile
		KOIs = Holczer_KOIs
		epochs = Holczer_epochs
		OCmin = Holczer_OCmin
		OCmin_err = Holczer_OCmin_err
		unique_KOIs = Holczer_unique_KOIs

	elif use_Holczer_or_gose == 'g':
		OCfile = TKS_OCfile
		KOIs = TKS_KOIs
		epochs = TKS_epochs
		OCmin = TKS_OCmin
		OCmin_err = TKS_OCmin_err
		unique_KOIs = TKS_unique_KOIs

	############# END LOADING TTVS ######################################


	################ LOAD THE CUMULATIVE KOI FILE FROM MAST #########################
	################### AND PERFORM SOME SIMPLE CALCULATIONS ON THEM ###################
	cumkois = ascii.read('/data/tethys/Documents/Software/MoonPy/cumkois_mast.txt')
	kepler_names = np.array(cumkois['kepler_name'])
	kepois = np.array(cumkois['kepoi_name'])
	kepoi_periods = np.array(cumkois['koi_period'])
	dispositions = np.array(cumkois['koi_disposition'])
	kepler_radius_rearth = np.array(cumkois['koi_prad'])
	kepler_radius_rearth_uperr = np.array(cumkois['koi_prad_err1'])
	kepler_radius_rearth_lowerr = np.array(cumkois['koi_prad_err2'])
	kepler_radius_rearth_err = np.nanmean((kepler_radius_rearth_uperr, np.abs(kepler_radius_rearth_lowerr)), axis=0)
	kepler_solar_mass = np.array(cumkois['koi_smass'])
	kepler_solar_mass_uperr = np.array(cumkois['koi_smass_err1'])
	kepler_solar_mass_lowerr = np.array(cumkois['koi_smass_err2'])
	kepler_solar_mass_err = np.nanmean((np.abs(kepler_solar_mass_uperr), np.abs(kepler_solar_mass_lowerr)), axis=0)
	FP_idxs = np.where(dispositions == 'FALSE POSITIVE')[0]
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

		#if (exclude_short_periods == 'y') and (float(this_planet_period) < 10): ##### the lower limit on your sims!
		#	print("SHORT PERIOD PLANET! SKIPPING.")
		#	continue 


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
	########### END CUMULATIVE KOI LOADING AND MANIPULATION ######################################




	#### LOAD THE CROSS-VALIDATION FILES (GENERATED IN THIS SAME SCRIPT, LATER) ######################
	try:
		########## LOAD THE CROSS-VALIDATION RESULTS (HOLCZER) ###############################
		Holczer_crossvalfile = pandas.read_csv('/data/tethys/Documents/Projects/NMoon_TTVs/Holczer_PTTV_results.csv')
		Holczer_cv_KOI = np.array(Holczer_crossvalfile['KOI']).astype(str) #### of the form 1.01
		Holczer_cv_PTTV_pcterr = np.array(Holczer_crossvalfile['PTTV_pcterr']).astype(float)
		Holczer_cv_ATTV_pcterr = np.array(Holczer_crossvalfile['ATTV_pcterr']).astype(float)
		Holczer_cv_phase_pcterr = np.array(Holczer_crossvalfile['phase_pcterr']).astype(float)
		Holczer_cv_good_idxs = []
		for i in np.arange(0,len(Holczer_cv_KOI),1):
			if (Holczer_cv_PTTV_pcterr[i] <= 5) and (Holczer_cv_ATTV_pcterr[i] <= 5) and (Holczer_cv_phase_pcterr[i] <= 5):
				Holczer_cv_good_idxs.append(i)
		loaded_holczer_crossvalfile = 'y'
		
	except:
		loaded_holczer_crossvalfile = 'n'

	try:
		#### LOAD THE CROSS-VALIDATION RESULTS (TKS) ########################################
		TKS_crossvalfile = pandas.read_csv('/data/tethys/Documents/Projects/NMoon_TTVs/GOSE_PTTV_results.csv')
		TKS_cv_KOI = np.array(TKS_crossvalfile['KOI']).astype(str) #### of the form 1.01
		TKS_cv_PTTV_pcterr = np.array(TKS_crossvalfile['PTTV_pcterr']).astype(float)
		TKS_cv_ATTV_pcterr = np.array(TKS_crossvalfile['ATTV_pcterr']).astype(float)
		TKS_cv_phase_pcterr = np.array(TKS_crossvalfile['phase_pcterr']).astype(float)
		TKS_cv_good_idxs = []
		for i in np.arange(0,len(TKS_cv_KOI),1):
			if (TKS_cv_PTTV_pcterr[i] <= 5) and (TKS_cv_ATTV_pcterr[i] <= 5) and (TKS_cv_phase_pcterr[i] <= 5):
				TKS_cv_good_idxs.append(i)
		loaded_TKS_crossvalfile = 'y'

	except:
		loaded_TKS_crossvalfile = 'n'

	try:
		#### LOAD THE CROSS-VALIDATION RESULTS (SIMULATIONS) #############################
		sim_crossvalfile = pandas.read_csv('/data/tethys/Documents/Projects/NMoon_TTVs/sim_PTTV_results.csv')
		sim_cv_KOI = np.array(sim_crossvalfile['sim']).astype(str) #### of the form 1.01
		sim_cv_PTTV_pcterr = np.array(sim_crossvalfile['PTTV_pcterr']).astype(float)
		sim_cv_ATTV_pcterr = np.array(sim_crossvalfile['ATTV_pcterr']).astype(float)
		sim_cv_phase_pcterr = np.array(sim_crossvalfile['phase_pcterr']).astype(float)
		sim_cv_good_idxs = []
		for i in np.arange(0,len(sim_cv_KOI),1):
			if (sim_cv_PTTV_pcterr[i] <= 5) and (sim_cv_ATTV_pcterr[i] <= 5) and (sim_cv_phase_pcterr[i] <= 5):
				sim_cv_good_idxs.append(i)
		loaded_sim_crossvalfile = 'y'

	except:
		loaded_sim_crossvalfile = 'n'





	##### FIND THE HADDEN & LITHWICK KOIs ###############################################################
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
	##### END PULLING OUT THE HADDEN & LITHWICK KOIS ###########################################



	############### LOAD FORECASTER RESULTS ###########################################################
	forecaster_MRfile = pandas.read_csv('/data/tethys/Documents/Central_Data/cumkoi_forecast_masses.csv')
	forecaster_KOIs = np.array(forecaster_MRfile['KOI']).astype(str)
	forecaster_mass_mearth = np.array(forecaster_MRfile['mass_mearth'])
	forecaster_mass_uperr = np.array(forecaster_MRfile['mass_mearth_uperr'])
	forecaster_mass_lowerr = np.array(forecaster_MRfile['mass_mearth_lowerr']) ### NEGATIVE NUMBER!!!
	#################### END LOAD FORECASTER RESULTS ###################################################





	##### GENERATE LISTS IN THE BIG FOR LOOP BELOW #########################################################

	radii = [] #### EARTH RADII
	radii_errors = []
	stellar_masses = []
	stellar_masses_errors = []
	P_TTVs = []
	P_plans = []
	deltaBICs = []
	Pip1_over_Pis = []
	Pi_over_Pim1s = []
	forecast_masses = []
	forecast_masses_uperr = []
	forecast_masses_lowerr = []

	TTV_amplitudes = []
	single_idxs = []
	multi_idxs = []
	in_HLcatalog_idxs = []
	notin_HLcatalog_idxs = []

	cv_PTTV_pcterrs = []
	cv_ATTV_pcterrs = []
	cv_phase_pcterrs = []




	######## PREPARE PTTV RESULTS FILES (CROSS-VALIDATION TESTS)
	if use_Holczer_or_gose == 'h':
		PTTV_resultsname = 'Holczer_PTTV_results.csv'
	elif use_Holczer_or_gose == 'g':
		PTTV_resultsname = 'GOSE_PTTV_results.csv'

	if cross_validate_LSPs == 'y':
		if os.path.exists('/data/tethys/Documents/Projects/NMoon_TTVs/'+PTTV_resultsname):
			### find last record KOI number:
			crossvalfile = pandas.read_csv('/data/tethys/Documents/Projects/NMoon_TTVs/'+PTTV_resultsname)
			cv_kepois_examined = np.array(crossvalfile['KOI']).astype(str)

		else:
			crossval_resultsfile = open('/data/tethys/Documents/Projects/NMoon_TTVs/'+PTTV_resultsname, mode='w')
			crossval_resultsfile.write('KOI,n_crossval_trials,n_epochs,n_removed,PTTV_median,PTTV_std,PTTV_skew,PTTV_kurtosis,PTTV_pcterr,ATTV_median,ATTV_std,ATTV_pcterr,phase_median,phase_std,phase_pcterr,deltaBIC,deltaBIC_std\n')
			crossval_resultsfile.close()
			cv_kepois_examined = np.array([])

	else:
		cv_kepois_examined = np.array([])
	##### END PREPARE PTTV RESULTS FILES (CROSS-VALIDATION TESTS)





	################### BIG ANALYSIS FOR LOOP ###################################
	entrynum = 0
	for nkepoi, kepoi in enumerate(kepois):

		this_planet_idx = np.where(kepoi == kepois)[0]
		this_planet_period = kepoi_periods[this_planet_idx][0]

		if (exclude_short_periods == 'y') and (float(this_planet_period) < 10): ##### the lower limit on your sims!
			print("SHORT PERIOD PLANET! SKIPPING.")
			continue 


		#if (cross_validate_LSPs == 'y') and (str(kepoi) in cv_kepois_examined):
		#	print('ALREADY EXAMINED. SKIPPING.')
		#	print(' ')
		#	continue

		if nkepoi in FP_idxs:
			print('Skipping false positive: ', kepoi)
			print(' ')
			continue


		##### FIND THE CROSS-VALIDATION INDICES FOR THIS TARGET
		if loaded_holczer_crossvalfile == 'y':
			cv_holczer_match_idx = np.where(kepoi == Holczer_cv_KOI)[0]
			holczer_cv_period_pct_error = Holczer_cv_PTTV_pcterr[cv_holczer_match_idx]
			holczer_cv_amplitude_pct_error = Holczer_cv_ATTV_pcterr[cv_holczer_match_idx]
			holczer_cv_phase_pct_error = Holczer_cv_phase_pcterr[cv_holczer_match_idx]

		if loaded_TKS_crossvalfile == 'y':
			cv_TKS_match_idx = np.where(kepoi == TKS_cv_KOI)[0]
			TKS_cv_period_pct_error = TKS_cv_PTTV_pcterr[cv_TKS_match_idx]
			TKS_cv_amplitude_pct_error = TKS_cv_ATTV_pcterr[cv_TKS_match_idx]
			TKS_cv_phase_pct_error = TKS_cv_phase_pcterr[cv_TKS_match_idx]




		#### FIND THE FORECASTER ENTRY FOR THIS OBJECT ##########################
		forecaster_idx = int(np.where(forecaster_KOIs == kepoi)[0])
		print('nkepoi, forecaster_idx = ', nkepoi, forecaster_idx)
		forecast_mass = forecaster_mass_mearth[forecaster_idx]
		forecast_mass_uperr = np.abs(forecaster_mass_uperr[forecaster_idx])
		forecast_mass_lowerr = np.abs(forecaster_mass_lowerr[forecaster_idx])

		try:
			kepoi_period = kepoi_periods[nkepoi]
			print('KOI-'+str(kepoi))
			KOI_idxs = np.where(KOIs == kepoi)[0]

			if len(KOI_idxs) == 0:
				print(' ')
				#### it's not in the catalog! Continue!
				continue

			KOI_epochs, KOI_OCs, KOI_OCerrs = np.array(epochs[KOI_idxs]).astype(int), np.array(OCmin[KOI_idxs]).astype(float), np.array(OCmin_err[KOI_idxs]).astype(float)
			KOI_rms = np.sqrt(np.nanmean(KOI_OCs**2))
			orig_KOI_rms = KOI_rms


			##### OUTLIER REJECTION #########################################################
			DBSCAN_vector = np.vstack((KOI_epochs, KOI_OCs)).T 
			db = DBSCAN(eps=5*np.nanmedian(KOI_OCerrs), min_samples=int(len(KOI_epochs)/5)).fit(KOI_OCs.reshape(-1,1))			
			labels = db.labels_ 
			outlier_idxs = np.where(labels == -1)[0]
			KOI_epochs, KOI_OCs, KOI_OCerrs = np.delete(KOI_epochs, outlier_idxs), np.delete(KOI_OCs, outlier_idxs), np.delete(KOI_OCerrs, outlier_idxs)
			KOI_rms = np.sqrt(np.nanmean(KOI_OCs**2))


			##### PERFORM A LOMB-SCARGLE PERIODOGRAM ON THE ENTIRE SAMPLE OF TRANSIT TIMES ########################
			LSperiods = np.logspace(np.log10(2), np.log10(500), 5000)
			LSfreqs = 1/LSperiods
			LSpowers = LombScargle(KOI_epochs, KOI_OCs, KOI_OCerrs).power(LSfreqs)
			peak_power_idx = np.nanargmax(LSpowers)
			peak_power_period = LSperiods[peak_power_idx]
			peak_power_freq = 1/peak_power_period

			### NOW FIT A SINUSOID -- HAVE TO DEFINE IT LIKE THIS TO UTILIZE curve_fit()
			def sinecurve(tvals, amplitude, phase):
				angfreq = 2 * np.pi * peak_power_freq
				sinewave = amplitude * np.sin(angfreq * tvals + phase)
				return sinewave

			#### NOW FIT THAT SUCKER
			popt, pcov = curve_fit(sinecurve, KOI_epochs, KOI_OCs, sigma=KOI_OCerrs, bounds=([0, -2*np.pi], [20*KOI_rms, 2*np.pi]))
			
			#### calculate BIC and deltaBIC -- USE ALL THE DATAPOINTS!
			BIC_flat = chisquare(KOI_OCs, np.linspace(0,0,len(KOI_OCs)),KOI_OCerrs) #k = 2
			BIC_curve = 2*np.log(len(KOI_OCs)) + chisquare(KOI_OCs, sinecurve(KOI_epochs, *popt), KOI_OCerrs)
			### we want BIC_curve to be SMALLER THAN BIC_flat, despite the penalty, for the SINE MODEL TO HOLD WATER.
			#### SO IF THAT'S THE CASE, AND WE DO BIC_curve - BIC_flat, then delta-BIC will be negative, which is what we want.
			deltaBIC = BIC_curve - BIC_flat 




			#### NOW WE'LL DO EXACTLY AS ABOVE, BUT WITH THE CROSS-VALIDATION REMOVALS.
			if cross_validate_LSPs == 'y':
				#### lists to be used for evaluating the robustness of the periodogram results.
				cv_best_periods = []
				cv_deltaBICs = []
				cv_amplitudes = []
				cv_phases = []
				cv_popts = []
				cv_pcovs = []

				ntoremove = int(cv_frac_to_leave_out*len(KOI_epochs))
				if ntoremove < 1:
					ntoremove = 1
					cv_ntrials_this_time = len(KOI_epochs)
				else:
					cv_ntrials_this_time = cv_ntrials

				for cv_trialnum in np.arange(0,cv_ntrials_this_time,1):
					if ntoremove == 1:
						idxs_to_leave_out = cv_trialnum ### make sure you leave out every point, one per trial
					else:
						idxs_to_leave_out = np.random.randint(low=0, high=len(KOI_epochs), size=ntoremove)	
					cv_KOI_epochs = np.delete(KOI_epochs, idxs_to_leave_out)
					cv_KOI_OCs = np.delete(KOI_OCs, idxs_to_leave_out)
					cv_KOI_OCerrs = np.delete(KOI_OCerrs, idxs_to_leave_out)

					cv_LSperiods = np.logspace(np.log10(2), np.log10(500), 5000)
					cv_LSfreqs = 1/cv_LSperiods
					cv_LSpowers = LombScargle(cv_KOI_epochs, cv_KOI_OCs, cv_KOI_OCerrs).power(cv_LSfreqs)
					cv_peak_power_idx = np.nanargmax(cv_LSpowers)
					cv_peak_power_period = cv_LSperiods[cv_peak_power_idx]
					cv_peak_power_freq = 1/cv_peak_power_period

					if cv_trialnum == 0:
						cv_LSpowers_stack = cv_LSpowers
					else:
						cv_LSpowers_stack = np.vstack((cv_LSpowers_stack, cv_LSpowers))


					def cv_sinecurve(tvals, amplitude, phase):
						angfreq = 2 * np.pi * cv_peak_power_freq
						sinewave = amplitude * np.sin(angfreq * tvals + phase)
						return sinewave


					#### NOW FIT THAT SUCKER
					cv_popt, cv_pcov = curve_fit(cv_sinecurve, cv_KOI_epochs, cv_KOI_OCs, sigma=cv_KOI_OCerrs, bounds=([0, -2*np.pi], [20*KOI_rms, 2*np.pi]))
					cv_popts.append(cv_popt)
					cv_pcovs.append(cv_pcov)
					cv_amplitudes.append(cv_popt[0])
					cv_phases.append(cv_popt[1])
					
					#### calculate BIC and deltaBIC -- USE ALL THE DATAPOINTS!
					cv_BIC_flat = chisquare(KOI_OCs, np.linspace(0,0,len(KOI_OCs)),KOI_OCerrs) #k = 2
					cv_BIC_curve = 2*np.log(len(KOI_OCs)) + chisquare(KOI_OCs, sinecurve(KOI_epochs, *cv_popt), KOI_OCerrs)
					cv_deltaBIC = cv_BIC_curve - cv_BIC_flat 

					cv_best_periods.append(cv_peak_power_period)
					cv_deltaBICs.append(cv_deltaBIC)

				#### now compute the median and std for period fits, and the same for the deltaBIC
				cv_best_periods, cv_deltaBICs = np.array(cv_best_periods), np.array(cv_deltaBICs)
				cv_period_skew, cv_period_kurtosis = skew(cv_best_periods), kurtosis(cv_best_periods)
				cv_amplitudes, cv_phases = np.array(cv_amplitudes), np.array(cv_phases)
				cv_best_period_median, cv_best_period_std = np.nanmedian(cv_best_periods), np.nanstd(cv_best_periods)
				cv_period_pct_error = cv_best_period_std / cv_best_period_median
				cv_deltaBICs_median, cv_deltaBICs_std = np.nanmedian(cv_deltaBICs), np.nanstd(cv_deltaBICs)
				cv_amplitude_median, cv_amplitude_std = np.nanmedian(cv_amplitudes), np.nanstd(cv_amplitudes)
				cv_amplitude_pct_error = cv_amplitude_std / cv_amplitude_median
				cv_phase_median, cv_phase_std = np.nanmedian(cv_phases), np.nanstd(cv_phases)
				#phase_pct_error = np.abs(cv_phase_std / cv_phase_median)
				cv_phase_pct_error = cv_phase_std / (2*np.pi) #### SHOULD NOT BE A FRACTION OF THE VALUE! IT SHOULD BE A FRACTION OF THE CIRCLE!!!

				print('PTTV = '+str(cv_best_period_median)+' +/- '+str(cv_best_period_std))
				print('PTTV pct error = ', str(cv_period_pct_error*100))
				print("PTTV Skew, Kurtosis = ", str(cv_period_skew), str(cv_period_kurtosis))
				print('ATTV = '+str(cv_amplitude_median)+' +/- '+str(cv_amplitude_std))
				print('ATTV pct error = ', str(cv_amplitude_pct_error*100))
				print('Phase = '+str(cv_phase_median)+' +/- '+str(cv_phase_std))
				print('Phase pct error = ', str(cv_phase_pct_error*100))
				print("deltaBIC = "+str(cv_deltaBICs_median)+' +/- '+str(cv_deltaBICs_std))
				print(' ')

				crossval_resultsfile = open('/data/tethys/Documents/Projects/NMoon_TTVs/'+PTTV_resultsname, mode='a')
				#crossval_resultsfile.write('KOI,n_crossval_trials,n_epochs,n_removed,PTTV_median,PTTV_std,PTTV_skew,PTTV_kurtosis,PTTV_pcterr,ATTV_median,ATTV_std,ATTV_pcterr,phase_median,phase_std,phase_pcterr,deltaBIC,deltaBIC_std\n')
				crossval_resultsfile.write(str(kepoi)+','+str(cv_ntrials_this_time)+','+str(len(KOI_epochs))+','+str(ntoremove)+','+str(cv_best_period_median)+','+str(cv_best_period_std)+','+str(cv_period_skew)+','+str(cv_period_kurtosis)+','+str(cv_period_pct_error*100)+','+str(cv_amplitude_median)+','+str(cv_amplitude_std)+','+str(cv_amplitude_pct_error*100)+','+str(cv_phase_median)+','+str(cv_phase_std)+','+str(cv_phase_pct_error*100)+','+str(cv_deltaBICs_median)+','+str(cv_deltaBICs_std)+'\n')
				crossval_resultsfile.close()

			if show_plots == 'y':
				#### THIS IS THE FULL DATA FIT FROM ABOVE -- NO LEAVING OUT DATA.
				KOI_epochs_interp = np.linspace(np.nanmin(KOI_epochs), np.nanmax(KOI_epochs), 1000)
				KOI_TTV_interp = sinecurve(KOI_epochs_interp, *popt)
				plt.scatter(KOI_epochs, KOI_OCs, facecolor='LightCoral', edgecolor='k', alpha=0.7, zorder=2)
				plt.errorbar(KOI_epochs, KOI_OCs, yerr=KOI_OCerrs, ecolor='k', fmt='none', zorder=1, alpha=0.2)
				plt.plot(KOI_epochs_interp, KOI_TTV_interp, color='k', linestyle='--', linewidth=2, alpha=0.2)

				if cross_validate_LSPs == 'y':
					for cv_popt in cv_popts:
						cv_KOI_TTV_interp = cv_sinecurve(KOI_epochs_interp, *cv_popt)				
						plt.plot(KOI_epochs_interp, cv_KOI_TTV_interp, color='k', linestyle='--', linewidth=2, alpha=0.2)
				
				plt.plot(KOI_epochs, np.linspace(0,0,len(KOI_epochs)), color='k', linestyle=':', alpha=0.5, zorder=0)
				plt.xlabel("epoch")
				plt.ylabel('O - C [min]')
				plt.title('KOI-'+str(kepoi))
				plt.show()



			####### END CROSS-VALIDATION TEST #################################################


			###### FOR PLANETS WITH DISCERNIBLE TTVs, INCLUDE THEM IN THESE LISTS. --
			######## SCREEN BY WHETHER OR NOT THEY MEET YOUR CROSS-VALIDATION STANDARD, AS WELL.
			if cross_validate_LSPs == 'y':
				if (cv_period_pct_error <= 5) and (cv_amplitude_pct_error <= 5) and (cv_phase_pct_error <=5):
					good_cv = 'y'
				else:
					good_cv = 'n'
			elif cross_validate_LSPs == 'n':
				if use_Holczer_or_gose == 'h':
					if (holczer_cv_period_pct_error <= 5) and (holczer_cv_amplitude_pct_error <= 5) and (holczer_cv_phase_pct_error <= 5):
						good_cv = 'y'
					else:
						good_cv = 'n'

				elif use_Holczer_or_gose == 'g':
					if (TKS_cv_period_pct_error <= 5) and (TKS_cv_amplitude_pct_error <= 5) and (TKS_cv_phase_pct_error <= 5):
						good_cv = 'y'
					else:
						good_cv = 'n'		


			if deltaBIC <= -2:
				print('GOOD delta-BIC.')
			if good_cv == 'y':
				print('Good cross-validation.')
			if (deltaBIC <= -2) and (good_cv == 'y'):
				print('including this system.')
				radii.append(kepler_radius_rearth[nkepoi])
				radii_errors.append(kepler_radius_rearth_err[nkepoi])
				stellar_masses.append(kepler_solar_mass[nkepoi])
				stellar_masses_errors.append(kepler_solar_mass_err[nkepoi])
				P_TTVs.append(peak_power_period)
				P_plans.append(kepoi_period)
				amplitude = np.nanmax(np.abs(BIC_curve))
				TTV_amplitudes.append(amplitude)
				Pip1_over_Pis.append(kepoi_multi_Pip1_over_Pis)
				Pi_over_Pim1s.append(kepoi_multi_Pi_over_Pim1s)
				forecast_masses.append(forecast_mass)
				forecast_masses_uperr.append(forecast_mass_uperr)
				forecast_masses_lowerr.append(forecast_mass_lowerr)
				deltaBICs.append(deltaBIC)
				if 'KOI-'+kepoi in HL_KOIs:
					in_HLcatalog_idxs.append(entrynum)
				else:
					notin_HLcatalog_idxs.append(entrynum)

				if kepoi_multi[nkepoi] == True:
					multi_idxs.append(entrynum)
				elif kepoi_multi[nkepoi] == False:
					single_idxs.append(entrynum)
	
				#### CROSS-VALIDATION LISTS
				if cross_validate_LSPs == 'y':
					cv_PTTV_pcterrs.append(cv_period_pct_error)
					cv_ATTV_pcterrs.append(cv_amplitude_pct_error)
					cv_phase_pcterrs.append(cv_phase_pct_error)

				elif (cross_validate_LSPs == 'n') and (use_Holczer_or_gose == 'h') and (loaded_holczer_crossvalfile == 'y'):
					cv_PTTV_pcterrs.append(holczer_cv_period_pct_error)
					cv_ATTV_pcterrs.append(holczer_cv_amplitude_pct_error)
					cv_phase_pcterrs.append(holczer_cv_phase_pct_error)

				elif (cross_validate_LSPs == 'n') and (use_Holczer_or_gose == 'g') and (loaded_TKS_crossvalfile == 'y'):
					cv_PTTV_pcterrs.append(TKS_cv_period_pct_error)
					cv_ATTV_pcterrs.append(TKS_cv_amplitude_pct_error)
					cv_phase_pcterrs.append(TKS_cv_phase_pct_error)		



				entrynum += 1 #### advance the number of entries.

			print(' ')


		except:
			traceback.print_exc()
			time.sleep(5)








	################## CONVERT LISTS TO ARRAYS ###################################
	multi_idxs = np.array(multi_idxs)
	single_idxs = np.array(single_idxs)
	print('# singles , # multis = ', len(single_idxs), len(multi_idxs))
	print('# in HL , # not in HL = ', len(in_HLcatalog_idxs), len(notin_HLcatalog_idxs))
	in_HLcatalog_idxs = np.array(in_HLcatalog_idxs)
	notin_HLcatalog_idxs = np.array(notin_HLcatalog_idxs)
	notin_HLcatalog_single_idxs = np.intersect1d(single_idxs, notin_HLcatalog_idxs)
	notin_HLcatalog_multi_idxs = np.intersect1d(multi_idxs, notin_HLcatalog_idxs)
	single_notHL_idxs = np.intersect1d(notin_HLcatalog_idxs, single_idxs)
	multi_notHL_idxs = np.intersect1d(notin_HLcatalog_idxs, multi_idxs)
	multi_HL_idxs = np.intersect1d(in_HLcatalog_idxs, multi_idxs) #### should be the same as in_HLcatalog_idxs
	TTV_amplitudes = np.array(TTV_amplitudes)
	forecast_masses, forecast_masses_uperr, forecast_masses_lowerr = np.array(forecast_masses), np.array(forecast_masses_uperr), np.array(forecast_masses_lowerr)
	#### replace all forecast_masses == 0.0 with np.nan!
	forecast_masses[np.where(forecast_masses == 0.0)[0]] = np.nan
	P_TTVs = np.array(P_TTVs)
	P_plans = np.array(P_plans)
	P_plans_minutes = P_plans * 24 * 60 
	radii = np.array(radii)
	radii_errors = np.array(radii_errors)
	stellar_masses = np.array(stellar_masses)
	stellar_masses_errors = np.array(stellar_masses_errors)
	stellar_masses_mearth = (stellar_masses * M_sun) / M_earth 
	cv_PTTV_pcterrs = np.array(cv_PTTV_pcterrs)
	cv_ATTV_pcterrs = np.array(cv_ATTV_pcterrs)
	cv_phase_pcterrs = np.array(cv_phase_pcterrs)
	deltaBICs = np.array(deltaBICs)


	#### CUT BASED ON A BETTER THAN 5% ERROR ON PERIOD, AMPLITUDE AND PHASE ACROSS ALL SOLUTIONS.
	good_PTTV_pcterr_idxs = np.where(cv_PTTV_pcterrs <= 5)[0]
	good_ATTV_pcterr_idxs = np.where(cv_ATTV_pcterrs <= 5)[0]
	good_phase_pcterr_idxs = np.where(cv_phase_pcterrs <= 5)[0]
	good_cv_pcterr_idxs = np.intersect1d(good_PTTV_pcterr_idxs, good_ATTV_pcterr_idxs)
	good_cv_pcterr_idxs = np.intersect1d(good_cv_pcterr_idxs, good_phase_pcterr_idxs)



	#### COMPUTE THE MINIMUM fraction of the Hill sphere.
	fmin = np.vectorize(fmin)
	fmins_vals = fmin(forecast_masses, stellar_masses_mearth.value, TTV_amplitudes, P_plans_minutes)
	fmins, fmin_vars = fmins_vals[0], fmins_vals[1:]
	possible_moon_fmin_idxs = np.where(fmins < 1.0)[0]
	impossible_moon_fmin_idxs = np.where(fmins >= 1.0)[0]
	highest_mass = np.nanmax(forecast_masses)
	normalized_masses = forecast_masses / highest_mass
	amplitudes_div_masses = TTV_amplitudes / normalized_masses









	#########################################
	## P L O T T I N G ######################
	#########################################


	#### PLOT fmins vs P_plans for the multis in and not in the HL2017 catalog.
	fig = plt.figure(figsize=(6,8))
	ax = plt.subplot(111)
	plt.scatter(P_plans[notin_HLcatalog_single_idxs], fmins[notin_HLcatalog_single_idxs], facecolor='green', edgecolor='k', s=20, alpha=0.5, label='single non-HL2017')
	plt.scatter(P_plans[notin_HLcatalog_multi_idxs], fmins[notin_HLcatalog_multi_idxs], facecolor='DodgerBlue', edgecolor='k', s=20, alpha=0.5, label='multi non-HL2017')
	plt.scatter(P_plans[in_HLcatalog_idxs], fmins[in_HLcatalog_idxs], facecolor='LightCoral', edgecolor='k', s=20, alpha=0.5, label='multi HL2017')
	plt.plot(np.linspace(np.nanmin(P_plans), np.nanmax(P_plans), 100), np.linspace(0.4895, 0.4895, 100), c='k', linestyle='--', linewidth=2)
	plt.fill_between(np.linspace(np.nanmin(P_plans), np.nanmax(P_plans), 100), 1e-5, 0.4895, color='green', alpha=0.1)
	plt.fill_between(np.linspace(np.nanmin(P_plans), np.nanmax(P_plans), 100), 0.4895, 1e5, color='red', alpha=0.1)
	plt.ylim(np.nanmin(fmins), np.nanmax(fmins))

	plt.xlim(np.nanmin(P_plans), np.nanmax(P_plans))
	plt.xlabel(r'$P_{\mathrm{P}}$ [days]')
	plt.ylabel(r'minimum fraction $R_{\mathrm{Hill}}$')
	plt.xscale('log')
	plt.yscale('log')
	plt.legend()
	plt.subplots_adjust(left=0.125, bottom=0.09, right=0.9, top=0.95, wspace=0.2, hspace=0.2)
	plt.show()


	#### PLOT fmins vs P_TTVs! for the multis in and not in the HL2017 catalog. DOES THIS SHOW ANYTHING?!
	fig = plt.figure(figsize=(6,8))
	ax = plt.subplot(111)
	plt.scatter(P_TTVs[notin_HLcatalog_single_idxs], fmins[notin_HLcatalog_single_idxs], facecolor='green', edgecolor='k', s=20, alpha=0.5, label='single non-HL2017')
	plt.scatter(P_TTVs[notin_HLcatalog_multi_idxs], fmins[notin_HLcatalog_multi_idxs], facecolor='DodgerBlue', edgecolor='k', s=20, alpha=0.5, label='multi non-HL2017')
	plt.scatter(P_TTVs[in_HLcatalog_idxs], fmins[in_HLcatalog_idxs], facecolor='LightCoral', edgecolor='k', s=20, alpha=0.5, label='multi HL2017')
	plt.plot(np.linspace(np.nanmin(P_TTVs), np.nanmax(P_TTVs), 100), np.linspace(0.4895, 0.4895, 100), c='k', linestyle='--', linewidth=2)
	plt.fill_between(np.linspace(np.nanmin(P_TTVs), np.nanmax(P_TTVs), 100), 1e-5, 0.4895, color='green', alpha=0.1)
	plt.fill_between(np.linspace(np.nanmin(P_TTVs), np.nanmax(P_TTVs), 100), 0.4895, 1e5, color='red', alpha=0.1)
	plt.ylim(np.nanmin(fmins), np.nanmax(fmins))

	plt.xlim(np.nanmin(P_plans), np.nanmax(P_plans))
	plt.xlabel(r'$P_{\mathrm{TTV}}$ [epochs]')
	plt.ylabel(r'minimum fraction $R_{\mathrm{Hill}}$')
	plt.xscale('log')
	plt.yscale('log')
	plt.legend()
	plt.subplots_adjust(left=0.125, bottom=0.09, right=0.9, top=0.95, wspace=0.2, hspace=0.2)
	plt.show()






	####### PLOT ATTV vs Pplan for single and multis
	fig = plt.figure(figsize=(6,8))
	ax = plt.subplot(111)
	plt.scatter(P_plans[notin_HLcatalog_single_idxs], TTV_amplitudes[notin_HLcatalog_single_idxs], facecolor='green', edgecolor='k', s=20, alpha=0.5, label='single non-HL2017')
	plt.scatter(P_plans[notin_HLcatalog_multi_idxs], TTV_amplitudes[notin_HLcatalog_multi_idxs], facecolor='DodgerBlue', edgecolor='k', s=20, alpha=0.5, label='multi non-HL2017')
	plt.scatter(P_plans[in_HLcatalog_idxs], TTV_amplitudes[in_HLcatalog_idxs], facecolor='LightCoral', edgecolor='k', s=20, alpha=0.5, label='multi HL2017')
	plt.xlabel(r'$P_{\mathrm{P}}$ [days]')
	plt.ylabel('TTV amplitude [s]')
	plt.yscale('log')
	plt.xscale('log')
	plt.legend()
	plt.subplots_adjust(left=0.125, bottom=0.09, right=0.9, top=0.95, wspace=0.2, hspace=0.2)
	plt.show()
	


	######### PLOT ATTV vs PTTV for singles and multis
	
	plt.scatter(TTV_amplitudes[notin_HLcatalog_single_idxs], P_TTVs[notin_HLcatalog_single_idxs], facecolor='green', edgecolor='k', s=20, alpha=0.5, label='single non-HL2017')
	plt.scatter(TTV_amplitudes[notin_HLcatalog_multi_idxs], P_TTVs[notin_HLcatalog_multi_idxs], facecolor='DodgerBlue', edgecolor='k', s=20, alpha=0.5, label='multi non-HL2017')
	plt.scatter(TTV_amplitudes[in_HLcatalog_idxs], P_TTVs[in_HLcatalog_idxs], facecolor='LightCoral', edgecolor='k', s=20, alpha=0.5, label='multi HL2017')
	plt.ylabel(r'$P_{\mathrm{TTV}}$ [epochs]')
	plt.xlabel('TTV amplitude [s]')
	plt.yscale('log')
	plt.xscale('log')
	plt.legend()
	plt.show()
	


	########### PLOT PTTV vs Pplan for singles and multis (not in HL2017)
	"""
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



	#### PTTV vs PPLAN BINS FOR HEATMAPS BELOW
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



	##### PTTV vs Pplan HEATMAP OF *EVERYTHING* -- SINGLES AND MULTIS###############
	"""
	TTV_Pplan_hist2d = np.histogram2d(P_plans, P_TTVs, bins=[xbins, ybins])
	plt.imshow(TTV_Pplan_hist2d[0].T, origin='lower', cmap=cm.coolwarm)
	plt.xticks(ticks=np.arange(0,len(xbins),5), labels=np.around(np.log10(xbins[::5]),2))
	plt.yticks(ticks=np.arange(0,len(ybins),5), labels=np.around(np.log10(ybins[::5]), 2))
	plt.xlabel(r'$\log_{10} \, P_{\mathrm{P}}$ [days]')
	plt.ylabel(r'$\log_{10} \, P_{\mathrm{TTV}}$ [epochs]')
	plt.tight_layout()
	plt.show()

	np.save('/data/tethys/Documents/Projects/NMoon_TTVs/mazeh_PTTV10-1500_Pplan2-100_20x20_heatmap.npy', TTV_Pplan_hist2d)


	######## SAME AS ABOVE, USING MATPLOTLIB HIST2D RATHER THAN NUMPY HIST2D.
	plt.figure(figsize=(6,6))
	heatmap = plt.hist2d(P_plans, P_TTVs, bins=[xbins, ybins], cmap='coolwarm')[0]
	plt.scatter(P_plans, P_TTVs, facecolor='w', edgecolor='k', s=5, alpha=0.3)
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel(r'$P_{\mathrm{P}}$ [days]')
	plt.ylabel(r'$P_{\mathrm{TTV}}$ [epochs]')
	#plt.title('Matplotlib 2D histogram')
	plt.show()
	"""


	#### PTTV vs PPLAN FOR SINGLES AND MULTIS
	"""
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

	np.save('/data/tethys/Documents/Projects/NMoon_TTVs/heatmap_single.npy', heatmap_single)	
	plt.show()	
	"""



	"""
	##### LOOK AT THE DISTRIBUTION FOR HL2017 SOURCES and Non-HL2017 sources
	#### PTTV vs Pplan HEATMAP FOR THOSE IN AND OUTSIDE HL2017
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
	plt.show()	
	"""



	##### LOOK AT AMPLITUDE VS PTTV FOR HL2017 SOURCES and Non-HL2017 sources
	"""
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
	"""



	"""
	##### LOOK AT AMPLITUDE VS PTTV FOR SINGLES AND MULTIs
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
	"""




	"""
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
	"""



	##### PLOT amplitude and PTTV as a function of the MULTI periods
	"""
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
	"""



	##### CREATE ONE GIANT PANEL -- (4x4) with 1) sims, 2) non-HL singles, 3) non-HL multis, 4) HL, and 
	##### Pplan, PTTV, Amplitudes, DeltaBICs

	#### NEW -- make it 6 x 4, to include histograms of 4

	#np.save(projectdir+'/sim_deltaBIC_list.npy', deltaBIC_list[good_BIC_stable_idxs])
	#np.save(projectdir+'/sim_PTTVs.npy', P_TTVs)
	#np.save(projectdir+'/sim_Pplans.npy', P_plans)




	##### LOAD SIMULATION VALUES.
	try:
		if sim_prefix == '':		
			sim_deltaBIC_list = np.load('/data/tethys/Documents/Projects/NMoon_TTVs/sim_deltaBIC_list.npy')
			sim_PTTVs = np.load('/data/tethys/Documents/Projects/NMoon_TTVs/sim_PTTVs.npy')
			sim_Pplans = np.load('/data/tethys/Documents/Projects/NMoon_TTVs/sim_Pplans.npy')
			sim_TTV_amplitudes = np.load('/data/tethys/Dcuments/Projects/NMoon_TTVs/sim_TTV_amplitudes.npy')
		else:
			sim_deltaBIC_list = np.load('/data/tethys/Documents/Projects/NMoon_TTVs/'+sim_prefix+'_sim_deltaBIC_list.npy')
			sim_PTTVs = np.load('/data/tethys/Documents/Projects/NMoon_TTVs/'+sim_prefix+'_sim_PTTVs.npy')
			sim_Pplans = np.load('/data/tethys/Documents/Projects/NMoon_TTVs/'+sim_prefix+'_sim_Pplans.npy')
			sim_TTV_amplitudes = np.load('/data/tethys/Dcuments/Projects/NMoon_TTVs/'+sim_prefix+'_sim_TTV_amplitudes.npy')


	except:
		if sim_prefix == '':
			sim_deltaBIC_list = np.load(projectdir+'/sim_deltaBIC_list.npy')
			sim_PTTVs = np.load(projectdir+'/sim_PTTVs.npy')
			sim_Pplans = np.load(projectdir+'/sim_Pplans.npy')
			sim_TTV_amplitudes = np.load(projectdir+'/sim_TTV_amplitudes.npy')
		else:
			sim_deltaBIC_list = np.load(projectdir+'/'+sim_prefix+'_sim_deltaBIC_list.npy')
			sim_PTTVs = np.load(projectdir+'/'+sim_prefix+'_sim_PTTVs.npy')
			sim_Pplans = np.load(projectdir+'/'+sim_prefix+'_sim_Pplans.npy')
			sim_TTV_amplitudes = np.load(projectdir+'/'+sim_prefix+'_sim_TTV_amplitudes.npy')

	#sim_PTTVs = np.load('/run/media/amteachey/Auddy_Akiti/Teachey/Nmoon_TTVs/sim_PTTVs.npy')
	#sim_Pplans = np.load('/run/media/amteachey/Auddy_Akiti/Teachey/Nmoon_TTVs/sim_Pplans.npy')

	##### NOW GRAB AS MANY sim_PTTV_results as you can.
	nsims = len(sim_deltaBIC_list)
	nsims_crossvaled = len(sim_cv_PTTV_pcterr)

	if nsims_crossvaled < nsims:
		sim_deltaBIC_list = sim_deltaBIC_list[:nsims_crossvaled]
		sim_PTTVs = sim_PTTVs[:nsims_crossvaled]
		sim_Pplans = sim_Pplans[:nsims_crossvaled]
		sim_TTV_amplitudes = sim_TTV_amplitudes[:nsims_crossvaled]
	elif nsims_crossvaled > nsims:
		sim_cv_PTTV_pcterr = sim_cv_PTTV_pcterr[:nsims]
		sim_cv_ATTV_pcterr = sim_cv_ATTV_pcterr[:nsims]
		sim_cv_phase_pcterr = sim_cv_phase_pcterr[:nsims]
	else:
		#### they're equal length -- don't need to do anything.	
		pass	

	sim_cv_good_idxs = []
	for i in np.arange(0,len(sim_cv_PTTV_pcterr),1):
		if (sim_cv_PTTV_pcterr[i] <= 5) and (sim_cv_ATTV_pcterr[i] <= 5) and (sim_cv_phase_pcterr[i] <= 5):
			sim_cv_good_idxs.append(i)
	sim_cv_good_idxs = np.array(sim_cv_good_idxs)

	#### SHOULD BE SAFE TO DO THIS -- WE'RE NOT PAIRING THEM UP WITH ANYTHING ELSE, SO MAKE THE CUT!
	sim_deltaBIC_list = sim_deltaBIC_list[sim_cv_good_idxs]
	sim_PTTVs = sim_PTTVs[sim_cv_good_idxs]
	try:
		sim_TTV_amplitudes = sim_TTV_amplitudes[sim_cv_good_idxs]
	except:
		pass
	sim_Pplans = sim_Pplans[sim_cv_good_idxs]

	sim_TTV_amplitudes_minutes = sim_TTV_amplitudes / 60



	#### PREPARE DATA FOR 6 x 4 PLOT.
	deltaBIC_lists = [sim_deltaBIC_list, deltaBICs[single_notHL_idxs], deltaBICs[multi_notHL_idxs], deltaBICs[multi_HL_idxs], deltaBICs[possible_moon_fmin_idxs], deltaBICs[impossible_moon_fmin_idxs]]
	deltaBIC_bins = np.linspace(-100,-2,20)
	
	PTTV_lists = [sim_PTTVs, P_TTVs[single_notHL_idxs], P_TTVs[multi_notHL_idxs], P_TTVs[multi_HL_idxs], P_TTVs[possible_moon_fmin_idxs], P_TTVs[impossible_moon_fmin_idxs]]
	PTTV_bins = np.logspace(np.log10(2), np.log10(100), 20)
	
	TTVamp_lists = [sim_TTV_amplitudes_minutes, TTV_amplitudes[single_notHL_idxs], TTV_amplitudes[multi_notHL_idxs], TTV_amplitudes[multi_HL_idxs], TTV_amplitudes[possible_moon_fmin_idxs], TTV_amplitudes[impossible_moon_fmin_idxs]]
	TTVamp_bins = np.logspace(0,6,20)

	#PTTV_over_Pplan_lists = [sim_PTTVs / sim_Pplans, P_TTVs[single_notHL_idxs]/P_plans[single_notHL_idxs], P_TTVs[multi_notHL_idxs]/P_plans[multi_notHL_idxs], P_TTVs[multi_HL_idxs] / P_plans[multi_HL_idxs]]
	#PTTV_over_Pplan_bins = np.logspace(0,3,20)

	Pplan_lists = [sim_Pplans, P_plans[single_notHL_idxs], P_plans[multi_notHL_idxs], P_plans[multi_HL_idxs], P_plans[possible_moon_fmin_idxs], P_plans[impossible_moon_fmin_idxs]]
	Pplan_bins = np.logspace(np.log10(1), np.log10(1500), 20)
	#amp_over_mass_lists = [sim_AoverM, amplitudes_div_masses[single_notHL_idxs], amplitudes_div_masses[multi_notHL_idxs], amplitudes_div_masses[multi_HL_idxs]]


	row_labels = ['moon sims', 'singles', 'multis', 'HL2017', 'possible', 'impossible']
	col_labels = [r'$P_{\mathrm{P}}$ [days]', r'$P_{\mathrm{TTV}}$ [epochs]', '$A_{\mathrm{TTV}}$ [min]', r'$\Delta$BIC']
	column_list_of_lists = [Pplan_lists, PTTV_lists, TTVamp_lists, deltaBIC_lists]
	bin_lists = [Pplan_bins, PTTV_bins, TTVamp_bins, deltaBIC_bins]
	axis_scales = ['log', 'log', 'log', 'linear']

	colors = cm.viridis(np.linspace(0,1,len(bin_lists)))	

	nrows = 6 #### sim, single, multi, HL
	ncols = 4 #### Pplan, PTTV, deltaBIC

	fig, ax = plt.subplots(nrows,ncols, figsize=(7,8)) ### might need to reverse this
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
				ax[row][col].set_ylabel(row_labels[row])

			if row == nrows-1:
				ax[row][col].set_xlabel(col_labels[col])

			if row != nrows-1:
				ax[row][col].set_xticklabels([])

	#plt.tight_layout()
	plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.95, 
                    wspace=0.5, 
                    hspace=0.15)

	plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/histogram_6x4_real_planets.pdf', dpi=300)

	plt.show()



	#### let's compute a bunch of KS values.
	#### DO IT FOR HL against ALL, and HL against MULTIS.
	#### values we will test are P_TTVs, deltaBICs, P_plans, 

	try:
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
	except:
		traceback.print_exc()






	fig, ax1 = plt.subplots(nrows=1,ncols=1)
	all_holczer_heatmap = ax1.hist2d(P_plans, P_TTVs, bins=[xbins, ybins], cmap='coolwarm', density=False)[0]
	ax1.set_xscale('log')
	ax1.set_yscale('log')
	ax1.set_title('Holczer 2016')
	plt.show()



	#### 6 panel heatmaps
	fig, ((ax1,ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3,ncols=2)
	sim_heatmap = ax1.hist2d(sim_Pplans, sim_PTTVs, bins=[xbins, ybins], cmap='coolwarm', density=False)[0]
	#ax1.scatter(sim_Pplans, sim_PTTVs, facecolor='w', edgecolor='k', alpha=0.2, s=10)
	ax1.set_xscale('log')	
	ax1.set_yscale('log')
	ax1.set_title('sims')
	
	single_heatmap = ax2.hist2d(P_plans[single_idxs], P_TTVs[single_idxs], bins=[xbins, ybins], cmap='coolwarm', density=False)[0]
	#ax2.scatter(P_plans[single_idxs], P_TTVs[single_idxs], facecolor='w', edgecolor='k', alpha=0.2, s=10)
	ax2.set_xscale('log')
	ax2.set_yscale('log')
	ax2.set_title('singles')

	multi_nonHL_heatmap = ax3.hist2d(P_plans[notin_HLcatalog_multi_idxs], P_TTVs[notin_HLcatalog_multi_idxs], bins=[xbins, ybins], cmap='coolwarm', density=False)[0]
	#ax3.scatter(P_plans[notin_HLcatalog_multi_idxs], P_TTVs[notin_HLcatalog_multi_idxs], facecolor='w', edgecolor='k', alpha=0.2, s=10)
	ax3.set_xscale('log')
	ax3.set_yscale('log')
	ax3.set_title('multi non-HL2017')

	multi_HL_heatmap = ax4.hist2d(P_plans[in_HLcatalog_idxs], P_TTVs[in_HLcatalog_idxs], bins=[xbins, ybins], cmap='coolwarm', density=False)[0]
	#ax4.scatter(P_plans[in_HLcatalog_idxs], P_TTVs[in_HLcatalog_idxs], facecolor='w', edgecolor='k', alpha=0.2, s=10)
	ax4.set_xscale('log')
	ax4.set_yscale('log')
	ax4.set_title('multi HL2017')

	possible_moon_heatmap = ax5.hist2d(P_plans[possible_moon_fmin_idxs], P_TTVs[possible_moon_fmin_idxs], bins=[xbins, ybins], cmap='coolwarm', density=False)[0]
	#ax5.scatter(P_plans[in_HLcatalog_idxs], P_TTVs[possible_moon_fmin_idxs], facecolor='w', edgecolor='k', alpha=0.2, s=10)
	ax5.set_xscale('log')
	ax5.set_yscale('log')
	ax5.set_title('possible moon')

	impossible_moon_heatmap = ax6.hist2d(P_plans[impossible_moon_fmin_idxs], P_TTVs[impossible_moon_fmin_idxs], bins=[xbins, ybins], cmap='coolwarm', density=False)[0]
	#ax4.scatter(P_plans[in_HLcatalog_idxs], P_TTVs[impossible_moon_fmin_idxs], facecolor='w', edgecolor='k', alpha=0.2, s=10)
	ax6.set_xscale('log')
	ax6.set_yscale('log')
	ax6.set_title('impossible moon')
	plt.subplots_adjust(left=0.125, bottom=0.09, right=0.9, top=0.95, wspace=0.2, hspace=0.2)
	plt.show()






	####### HEATMAPS NORMALIZED BY NUMBER OF SYSTEMS IN THE PERIOD BIN!
	all_holczer_heatmap_normalized = np.zeros(shape=all_holczer_heatmap.shape)
	sim_heatmap_normalized = np.zeros(shape=sim_heatmap.shape)
	single_heatmap_normalized = np.zeros(shape=single_heatmap.shape)
	multi_nonHL_heatmap_normalized = np.zeros(shape=multi_nonHL_heatmap.shape)
	multi_HL_heatmap_normalized = np.zeros(shape=multi_HL_heatmap.shape)
	possible_moon_heatmap_normalized = np.zeros(shape=possible_moon_heatmap.shape)
	impossible_moon_heatmap_normalized = np.zeros(shape=impossible_moon_heatmap.shape)

	all_holczer_heatmap_frac_of_Pplan = np.zeros(shape=all_holczer_heatmap.shape)
	sim_heatmap_frac_of_Pplan = np.zeros(shape=sim_heatmap.shape)
	single_heatmap_frac_of_Pplan = np.zeros(shape=single_heatmap.shape)
	multi_nonHL_heatmap_frac_of_Pplan = np.zeros(shape=multi_nonHL_heatmap.shape)
	multi_HL_heatmap_frac_of_Pplan = np.zeros(shape=multi_HL_heatmap.shape)
	possible_moon_heatmap_frac_of_Pplan = np.zeros(shape=possible_moon_heatmap.shape)
	impossible_moon_heatmap_frac_of_Pplan = np.zeros(shape=impossible_moon_heatmap.shape)


	nrows, ncols = sim_heatmap.shape

	for col in np.arange(0,ncols,1):
		all_holczer_heatmap_normalized[col] = (all_holczer_heatmap[col] - np.nanmin(all_holczer_heatmap[col])) / (np.nanmax(all_holczer_heatmap[col]) - np.nanmin(all_holczer_heatmap[col]))
		sim_heatmap_normalized[col] = (sim_heatmap[col] - np.nanmin(sim_heatmap[col])) / (np.nanmax(sim_heatmap[col]) - np.nanmin(sim_heatmap[col]))
		single_heatmap_normalized[col] = (single_heatmap[col] - np.nanmin(single_heatmap[col])) / (np.nanmax(single_heatmap[col]) - np.nanmin(single_heatmap[col]))
		multi_nonHL_heatmap_normalized[col] = (multi_nonHL_heatmap[col] - np.nanmin(multi_nonHL_heatmap[col])) / (np.nanmax(multi_nonHL_heatmap[col]) - np.nanmin(multi_nonHL_heatmap[col]))
		multi_HL_heatmap_normalized[col] = (multi_HL_heatmap[col] - np.nanmin(multi_HL_heatmap[col])) / (np.nanmax(multi_HL_heatmap[col]) - np.nanmin(multi_HL_heatmap[col]))
		possible_moon_heatmap_normalized[col] = (possible_moon_heatmap[col] - np.nanmin(possible_moon_heatmap[col])) / (np.nanmax(possible_moon_heatmap[col]) - np.nanmin(possible_moon_heatmap[col]))
		impossible_moon_heatmap_normalized[col] = (impossible_moon_heatmap[col] - np.nanmin(impossible_moon_heatmap[col])) / (np.nanmax(impossible_moon_heatmap[col]) - np.nanmin(impossible_moon_heatmap[col]))



		all_holczer_heatmap_frac_of_Pplan[col] = all_holczer_heatmap[col] / np.nansum(all_holczer_heatmap[col]) #
		sim_heatmap_frac_of_Pplan[col] = sim_heatmap[col] / np.nansum(sim_heatmap[col]) 
		single_heatmap_frac_of_Pplan[col] = single_heatmap[col] / np.nansum(single_heatmap[col]) 
		multi_nonHL_heatmap_frac_of_Pplan[col] = multi_nonHL_heatmap[col] / np.nansum(multi_nonHL_heatmap[col]) 
		multi_HL_heatmap_frac_of_Pplan[col] = multi_HL_heatmap[col] / np.nansum(multi_HL_heatmap[col]) 
		possible_moon_heatmap_frac_of_Pplan[col] = possible_moon_heatmap[col] / np.nansum(possible_moon_heatmap[col]) 
		impossible_moon_heatmap_frac_of_Pplan[col] = impossible_moon_heatmap[col] / np.nansum(impossible_moon_heatmap[col])



	##### PLOT THE NORMALIZED VERSION		
	colormap_choice = 'GnBu'
	fig, ((ax1,ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3,ncols=2, sharex=True, sharey=True, figsize=(6,8))
	#### upper left
	ax1.imshow(np.nan_to_num(sim_heatmap_normalized.T), origin='lower', aspect='auto', cmap=colormap_choice)
	ax1.set_ylabel(r'$\log_{10} \, P_{\mathrm{TTV}}$ [epochs]')
	ax1.set_yticks(np.arange(0,19,1)[::4])
	ax1.set_yticklabels(np.around(np.log10(ycenters[::4]), 2))
	#ax1.text(10**2.8, 10**1.8, 'Sims')

	#### upper right
	ax2.imshow(np.nan_to_num(single_heatmap_normalized.T), origin='lower', aspect='auto', cmap=colormap_choice)

	#### middle left
	ax3.imshow(np.nan_to_num(multi_nonHL_heatmap_normalized.T), origin='lower', aspect='auto', cmap=colormap_choice)
	ax3.set_ylabel(r'$\log_{10} \, P_{\mathrm{TTV}}$ [epochs]')
	#ax3.set_xlabel(r'$\log_{10} \, P_{\mathrm{P}}$ [days]')
	ax3.set_xticks(np.arange(0,19,1)[::4])
	ax3.set_xticklabels(np.around(np.log10(xcenters[::4]),2))
	ax3.set_yticks(np.arange(0,19,1)[::4])
	ax3.set_yticklabels(np.around(np.log10(ycenters[::4]),2))

	#### middle right
	ax4.imshow(np.nan_to_num(multi_HL_heatmap_normalized.T), origin='lower', aspect='auto', cmap=colormap_choice)
	#ax4.set_xlabel(r'$\log_{10} \, P_{\mathrm{P}}$ [days]')
	ax4.set_xticks(np.arange(0,19,1)[::4])
	ax4.set_xticklabels(np.around(np.log10(xcenters[::4]),2))

	#### lower left
	ax5.imshow(np.nan_to_num(possible_moon_heatmap_normalized.T), origin='lower', aspect='auto', cmap=colormap_choice)
	ax5.set_ylabel(r'$\log_{10} \, P_{\mathrm{TTV}}$ [epochs]')
	ax5.set_xlabel(r'$\log_{10} \, P_{\mathrm{P}}$ [days]')
	ax5.set_xticks(np.arange(0,19,1)[::4])
	ax5.set_xticklabels(np.around(np.log10(xcenters[::4]),2))
	ax5.set_yticks(np.arange(0,19,1)[::4])
	ax5.set_yticklabels(np.around(np.log10(ycenters[::4]),2))

	#### lower right
	ax6.imshow(np.nan_to_num(impossible_moon_heatmap_normalized.T), origin='lower', aspect='auto', cmap=colormap_choice)
	ax6.set_xlabel(r'$\log_{10} \, P_{\mathrm{P}}$ [days]')
	ax6.set_xticks(np.arange(0,19,1)[::4])
	ax6.set_xticklabels(np.around(np.log10(xcenters[::4]),2))

	plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.95, 
                    wspace=0.15, 
                    hspace=0.15)

	plt.show()






	#### PLOT THE FRACTION OF THE COLUMN VERSION
	colormap_choice = 'GnBu'
	fig, ((ax1,ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3,ncols=2, sharex=True, sharey=True, figsize=(6,8))
	#### upper left
	ax1.imshow(np.nan_to_num(sim_heatmap_frac_of_Pplan.T), vmax=0.5, origin='lower', aspect='auto', cmap=colormap_choice)
	ax1.set_ylabel(r'$\log_{10} \, P_{\mathrm{TTV}}$ [epochs]')
	ax1.set_yticks(np.arange(0,19,1)[::4])
	ax1.set_yticklabels(np.around(np.log10(ycenters[::4]), 2))
	#ax1.text(10**2.8, 10**1.8, 'Sims')

	#### upper right
	ax2.imshow(np.nan_to_num(single_heatmap_frac_of_Pplan.T),vmax=0.5, origin='lower', aspect='auto', cmap=colormap_choice)

	#### middle left
	ax3.imshow(np.nan_to_num(multi_nonHL_heatmap_frac_of_Pplan.T),vmax=0.5, origin='lower', aspect='auto', cmap=colormap_choice)
	ax3.set_ylabel(r'$\log_{10} \, P_{\mathrm{TTV}}$ [epochs]')
	#ax3.set_xlabel(r'$\log_{10} \, P_{\mathrm{P}}$ [days]')
	ax3.set_xticks(np.arange(0,19,1)[::4])
	ax3.set_xticklabels(np.around(np.log10(xcenters[::4]),2))
	ax3.set_yticks(np.arange(0,19,1)[::4])
	ax3.set_yticklabels(np.around(np.log10(ycenters[::4]),2))

	#### middle right
	ax4.imshow(np.nan_to_num(multi_HL_heatmap_frac_of_Pplan.T), vmax=0.5, origin='lower', aspect='auto', cmap=colormap_choice)
	#ax4.set_xlabel(r'$\log_{10} \, P_{\mathrm{P}}$ [days]')
	ax4.set_xticks(np.arange(0,19,1)[::4])
	ax4.set_xticklabels(np.around(np.log10(xcenters[::4]),2))

	#### lower left
	ax5.imshow(np.nan_to_num(possible_moon_heatmap_frac_of_Pplan.T), vmax=0.5, origin='lower', aspect='auto', cmap=colormap_choice)
	ax5.set_ylabel(r'$\log_{10} \, P_{\mathrm{TTV}}$ [epochs]')
	ax5.set_xlabel(r'$\log_{10} \, P_{\mathrm{P}}$ [days]')
	ax5.set_xticks(np.arange(0,19,1)[::4])
	ax5.set_xticklabels(np.around(np.log10(xcenters[::4]),2))
	ax5.set_yticks(np.arange(0,19,1)[::4])
	ax5.set_yticklabels(np.around(np.log10(ycenters[::4]),2))

	#### lower right
	ax6.imshow(np.nan_to_num(impossible_moon_heatmap_frac_of_Pplan.T), vmax=0.5, origin='lower', aspect='auto', cmap=colormap_choice)
	ax6.set_xlabel(r'$\log_{10} \, P_{\mathrm{P}}$ [days]')
	ax6.set_xticks(np.arange(0,19,1)[::4])
	ax6.set_xticklabels(np.around(np.log10(xcenters[::4]),2))

	plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.95, 
                    wspace=0.15, 
                    hspace=0.15)

	plt.show()







	def normalizer(values):
		return (values - np.nanmin(values)) / (np.nanmax(values) - np.nanmin(values))






	##### DO THE PROBABILITY MATH 
	#### UNIFORM P(moon | Pp):
	Pmoon_giv_PTTV_Pplan_uniform = sim_heatmap_frac_of_Pplan.T / all_holczer_heatmap_frac_of_Pplan.T

	#### propto period
	Pmoon_giv_PTTV_Pplan_propto_P = (sim_heatmap_frac_of_Pplan.T * xcenters) / all_holczer_heatmap_frac_of_Pplan.T

	#### propto P^2
	Pmoon_giv_PTTV_Pplan_propto_P2 = (sim_heatmap_frac_of_Pplan.T * xcenters**2) / all_holczer_heatmap_frac_of_Pplan.T

	#### propto 1/P
	Pmoon_giv_PTTV_Pplan_propto_1oP = (sim_heatmap_frac_of_Pplan.T * (1/xcenters)) / all_holczer_heatmap_frac_of_Pplan.T


	#### LET'S LOOK AT JUST THESE

	a = Pmoon_giv_PTTV_Pplan_uniform
	b = Pmoon_giv_PTTV_Pplan_propto_P
	c = Pmoon_giv_PTTV_Pplan_propto_P2
	d = Pmoon_giv_PTTV_Pplan_propto_1oP


	fig = plt.figure(figsize=(6,6))
	ax = plt.subplot(111)
	ax.imshow(Pmoon_giv_PTTV_Pplan_uniform, origin='lower', aspect='auto')
	ax.set_ylabel(r'$\log_{10} \, P_{\mathrm{TTV}}$ [epochs]')
	ax.set_xlabel(r'$\log_{10} \, P_{\mathrm{P}}$ [days]')
	ax.set_xticks(np.arange(0,19,1)[::4])
	ax.set_xticklabels(np.around(np.log10(xcenters[::4]),2))
	ax.set_yticks(np.arange(0,19,1)[::4])
	ax.set_yticklabels(np.around(np.log10(ycenters[::4]),2))
	plt.subplots_adjust(left=0.125,
                   	bottom=0.1, 
                    right=0.9, 
                    top=0.95, 
                    wspace=0.15, 
                    hspace=0.15)
	plt.show()



	#### now plot them

	colormap_choice = 'GnBu'
	fig, ((ax1,ax2), (ax3, ax4)) = plt.subplots(nrows=2,ncols=2, sharex=True, sharey=True, figsize=(8,8))
	#### upper left
	ax1.imshow(Pmoon_giv_PTTV_Pplan_uniform, origin='lower', aspect='auto', cmap=colormap_choice)
	ax1.set_ylabel(r'$\log_{10} \, P_{\mathrm{TTV}}$ [epochs]')
	ax1.set_yticks(np.arange(0,19,1)[::4])
	ax1.set_yticklabels(np.around(np.log10(ycenters[::4]), 2))
	#ax1.text(10**2.8, 10**1.8, 'Sims')

	#### upper right
	ax2.imshow(Pmoon_giv_PTTV_Pplan_propto_1oP, origin='lower', aspect='auto', cmap=colormap_choice)

	#### lower left
	ax3.imshow(Pmoon_giv_PTTV_Pplan_propto_P, origin='lower', aspect='auto', cmap=colormap_choice)
	ax3.set_ylabel(r'$\log_{10} \, P_{\mathrm{TTV}}$ [epochs]')
	ax3.set_xlabel(r'$\log_{10} \, P_{\mathrm{P}}$ [days]')
	ax3.set_xticks(np.arange(0,19,1)[::4])
	ax3.set_xticklabels(np.around(np.log10(xcenters[::4]),2))
	ax3.set_yticks(np.arange(0,19,1)[::4])
	ax3.set_yticklabels(np.around(np.log10(ycenters[::4]),2))

	#### lower right
	ax4.imshow(Pmoon_giv_PTTV_Pplan_propto_P2, origin='lower', aspect='auto', cmap=colormap_choice)
	ax4.set_xlabel(r'$\log_{10} \, P_{\mathrm{P}}$ [days]')
	ax4.set_xticks(np.arange(0,19,1)[::4])
	ax4.set_xticklabels(np.around(np.log10(xcenters[::4]),2))

	plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.95, 
                    wspace=0.15, 
                    hspace=0.15)

	plt.show()



















except:
	traceback.print_exc()



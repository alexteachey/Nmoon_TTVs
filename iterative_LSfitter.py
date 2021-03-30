from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas
import os
import traceback
from astropy.timeseries import LombScargle
from scipy.optimize import curve_fit
import time
import pickle


show_debug_plots = input('Do you want to show debug plots? y/n: ')

try:

	def chisquare(data,model,error):
		return np.nansum(((data-model)**2 / (error**2)))

	def BIC(nparams,data,model,error):
		return nparams*np.log(len(data)) + chisquare(data,model,error)

	def LS_to_deltaBIC(pmin, pmax, xvals, yvals, yerrs, nmoons=None):
		#### FIT THE LOMB SCARGLE
		### if yvals do not have a median of zero, make it so
		median_yvals = np.nanmedian(yvals)
		if median_yvals < 0:
			#### need a bost.
			yvals = yvals + np.abs(median_yvals)
		elif median_yvals > 0:
			yvals = yvals - np.abs(median_yvals)

		period_min = pmin
		period_max = pmax
		LS_frequencies = np.logspace(np.log10(1/period_max), np.log10(1/period_min), 5000)
		LS_periods = 1 / LS_frequencies
		LS_powers = LombScargle(epochs, TTVobs, TTVerrs).power(LS_frequencies)
		yvals_rms = np.sqrt(np.nanmean(yvals**2))
		est_amplitude = np.sqrt(2)*yvals_rms
		### scale LS_powers by the amplitude
		LS_powers = LS_powers * est_amplitude 
		best_LS_freq = LS_frequencies[np.argmax(LS_powers)]
		best_LS_period = 1/best_LS_freq
		best_LS_angfreq = 2*np.pi*best_LS_freq
		print('best_LS_period = ', best_LS_period)

		def sinecurve(xvals, amplitude, phase):
			#### the frequency must be supplied.
			return amplitude * np.sin((best_LS_angfreq*xvals) + phase)

		#### do the curve fit
		popt, pcov = curve_fit(sinecurve, xvals, yvals, sigma=yerrs, bounds=([0,-2*np.pi],[2*np.nanmax(LS_powers),2*np.pi]))
		BIC_flat = chisquare(data=yvals, model=np.linspace(0,0,len(xvals)), error=yerrs) #k = 2
		BIC_curve = 2*np.log(len(TTVobs)) + chisquare(data=yvals, model=sinecurve(xvals, *popt), error=yerrs)
		deltaBIC = BIC_curve - BIC_flat 
		print("BIC_flat = ", BIC_flat)
		print("BIC_curve = ", BIC_curve)

		#### subtact off the CURVE!
		new_yvals = yvals - sinecurve(xvals, *popt)
		assert len(new_yvals) == len(yvals)

		if show_debug_plots == 'y':
			fig, (ax1, ax2) = plt.subplots(2, figsize=(8,8))
			ax1.scatter(xvals, yvals, s=20, facecolor='LightCoral', edgecolor='k', zorder=1)
			ax1.errorbar(xvals, yvals, yerr=yerrs, ecolor='k', zorder=0, fmt='none')
			xsmooth = np.linspace(np.nanmin(xvals), np.nanmax(xvals), 1000)
			ax1.plot(xsmooth, sinecurve(xsmooth, *popt), linestyle='--', color='k', linewidth=2, zorder=2)
			ax1.set_xlabel('epoch')
			ax1.set_ylabel('O - C')
			try:
				ax1.set_title('sim '+str(sim)+', # moons = '+str(nmoons))
			except:
				ax1.set_title('sim '+str(sim))
			ax2.plot(LS_periods, LS_powers)

			ax2.set_xscale('log')
			ax2.set_ylabel('Power')
			ax2.set_xlabel('Period [epochs]')
			plt.show()


		return new_yvals, deltaBIC, best_LS_period, LS_periods, LS_powers


	projectdir = '/run/media/amteachey/Auddy_Akiti/Teachey/Nmoon_TTVs'
	sim_prefix = input('What is the sim prefix? (blank for current): ')
	if len(sim_prefix) == 0:
		pass

	elif sim_prefix[-1] != '_':
		sim_prefix = sim_prefix+'_'

	if 'RUN4' in sim_prefix:
		TTVdir = projectdir+'/'+sim_prefix+'FIXED_sim_TTVs'
	else:
		TTVdir = projectdir+'/'+sim_prefix+'sim_TTVs'

	simsum = pandas.read_csv(projectdir+'/'+sim_prefix+'simulation_summary.csv')
	sims = np.array(simsum['sim']).astype(str)
	nmoons = np.array(simsum['nmoons']).astype(int)
	modeldictdir = projectdir+'/'+sim_prefix+'sim_model_settings'


	niterations_list = []

	for nsim,sim in enumerate(sims):

		best_BIC = -np.inf 

		sim_nmoons = nmoons[nsim]
		TTVfile = 'TTVsim'+sim+'_TTVs.csv'
		TTVs = pandas.read_csv(TTVdir+'/'+TTVfile)
		epochs = np.array(TTVs['epoch']).astype(int)
		TTVobs = np.array(TTVs['TTVob']).astype(float) ### seconds
		TTVerrs = np.array(TTVs['timing_error']).astype(float) ### seconds
		sim_model_dict = pickle.load(open(modeldictdir+'/TTVsim'+str(sim)+'_system_dictionary.pkl', "rb"))
		sim_moon_keys = ['I', 'II', 'III', 'IV', 'V'][:sim_nmoons] ### if you have 3 moons, this will return 'I', 'II', 'III'.
		sim_moon_periods_days = []
		for smk in sim_moon_keys:
			sim_moon_period_seconds = sim_model_dict[smk]['P'] ### seconds
			sim_moon_period_days = sim_moon_period_seconds / (60 * 60 * 24)
		sim_moon_periods_days.append(sim_moon_period_days)

		#### NOW WE'RE GOING TO ITERATE ON HOW MANY THE NUMBER OF CYCLES WE CAN REMOVE, AND SEE HOW THAT LINES UP WITH THE NUMBER OF MOONS
		
		#### FIRST TIME COMPUTE IT... SEE HOW WE GO.

		niterations = 0
		new_TTVobs, new_deltaBIC, new_best_period, LSperiods, LSpowers = LS_to_deltaBIC(pmin=2, pmax=500, xvals=epochs, yvals=TTVobs, yerrs=TTVerrs, nmoons=sim_nmoons)
		deltaBIC_list = [new_deltaBIC]
		
		print(' ')
		print(' X X X X X X X X X X ')
		print('sim '+str(sim))
		print("deltaBIC_list: ", deltaBIC_list)	
		print('# moons: ', sim_nmoons)

		if new_best_period == 2.0:
			print('bad solution (PTTV precisely 2.0).')

		while deltaBIC_list[-1] < 0:
			niterations += 1
			new_TTVobs, new_deltaBIC, new_best_period, new_periods, new_powers = LS_to_deltaBIC(pmin=2, pmax=500, xvals=epochs, yvals=new_TTVobs, yerrs=TTVerrs, nmoons=sim_nmoons)
			deltaBIC_list.append(new_deltaBIC)
			#### this loop concludes when deltaBIC_list[-1] >= 0
			if new_best_period == 2.0:
				break

		niterations_list.append(niterations)
		print("deltaBIC_list: ", deltaBIC_list)	
		print('# of deltaBIC improvements: ', niterations)
		#print(' X X X X X X X X X X ')
		print(' ')
		if show_debug_plots != 'y':
			time.sleep(5)




except:
	traceback.print_exc()








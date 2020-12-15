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




def sinewave(tvals, frequency, offset):
	return amplitude * np.sin(2*np.pi*frequency*tvals + offset)


def chisquare(data,model,error):
	return np.nansum(((data-model)**2 / (error**2)))

def BIC(nparams,data,model,error):
	return nparams*np.log(len(data)) + chisquare(data,model,error)







##### this script will accept a REBOUND simulation number and plot everything for that system

try:
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


	summaryfile = pandas.read_csv(projectdir+'/simulation_summary.csv')
	sumfile_cols = summaryfile.columns		

	plot_individual = input('Do you want to plot an individual simulation? y/n: ')
	if plot_individual == 'y':
		sim_num = int(input('What is the simulation number you want to see? '))
		sims = np.array([sim_num])
		show_individual_plots = 'y' 


	else:
		sims = np.array(summaryfile['sim'])
		show_individual_plots = input('Do you want to show individual system plots? y/n: ')
		sim_obs_summary = open(projectdir+'/simulated_observations.csv', mode='w')
		sim_obs_summary.write('sim,Pplan_days,ntransits,TTV_rmsamp_sec,TTVperiod_epochs,peak_power,fit_sineamp,deltaBIC\n')
		sim_obs_summary.close()



	#### NEW SUMMARY STATISTICS LISTS
	deltaBIC_list = []
	peak_power_periods_list = []
	nmoons = np.array(summaryfile['nmoons'])
	megno_vals = np.array(summaryfile['MEGNO'])
	spockprobs = np.array(summaryfile['SPOCK_survprop'])

	for sim in sims:
		try:
			print('sim # '+str(sim))
			#### MODEL INPUTS
			sim_model_dict = pickle.load(open(modeldictdir+'/TTVsim'+str(sim)+'_system_dictionary.pkl', "rb"))
			
			#### TTV file
			sim_TTVs = pandas.read_csv(ttvfiledir+'/TTVsim'+str(sim)+'_TTVs.csv')
			sim_TTV_epochs = np.array(sim_TTVs['epoch']).astype(int)
			sim_TTV_OminusC = np.array(sim_TTVs['TTVob']).astype(float)
			sim_TTV_errors = np.array(sim_TTVs['timing_error']).astype(float)


			#### PERIODOGRAM OF THE TTVs
			sim_periodogram = np.load(LSdir+'/TTVsim'+str(sim)+'_periodogram.npy')
			
			sim_xpos = np.load(positionsdir+'/TTVsim'+str(sim)+'_xpos.npy')
			sim_ypos = np.load(positionsdir+'/TTVsim'+str(sim)+'_ypos.npy')
			

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
			peak_power_periods_list.append(peak_power_period)

			if show_individual_plots == 'y':
				#### plot the periodogram
				plt.plot(periods, powers, c='DodgerBlue', linewidth=2, alpha=0.7)
				plt.xlabel('Period [epochs]')
				plt.ylabel('Power')
				plt.xscale('log')
				plt.title('best period = '+str(round(peak_power_period, 3))+' epochs')
				plt.show()




			#### perform a curve_fit to find amplitude and offset, to be plotted below (and for Delta-BIC -- or just delta X^2)?
			frequency = 1/peak_power_period
			def sinewave(tvals, amplitude, offset):
				return amplitude * np.sin(2*np.pi*frequency*tvals + offset)

			popt, pcov = curve_fit(sinewave, sim_TTV_epochs, sim_TTV_OminusC, sigma=sim_TTV_errors, bounds=([0, -2*np.pi], [10*sim_TTV_rmsamp, 2*np.pi]))
			interp_epochs = np.linspace(np.nanmin(sim_TTV_epochs), np.nanmax(sim_TTV_epochs), 1000)
			sinecurve = sinewave(interp_epochs, *popt)

			BIC_flat = BIC(nparams=0, data=sim_TTV_OminusC, model=np.linspace(0,0,len(sim_TTV_OminusC)), error=sim_TTV_errors)
			BIC_curve = BIC(nparams=2, data=sim_TTV_OminusC, model=sinewave(sim_TTV_epochs,*popt), error=sim_TTV_errors)

			#### we want Delta-BIC to be negative to indicate an improvement!
			###### Now, BIC_curve is an improvement over BIC_flat if BIC_curve < BIC_flat (even with extra complexity, the model is improved)
			####### so let deltaBIC = BIC_curve - BIC_flat: if BIC_curve is indeed < BIC_flat, then deltaBIC will be negative. SO:
			deltaBIC = BIC_curve - BIC_flat
			deltaBIC_list.append(deltaBIC)


			if show_individual_plots == 'y':
				#### plot the TTVs -- and fit a best fitting SINUSOID BASED ON THE PERIODOGRAM PERIOD?
				plt.scatter(sim_TTV_epochs, sim_TTV_OminusC, facecolor='LightCoral', edgecolor='k', s=20, zorder=2)
				plt.errorbar(sim_TTV_epochs, sim_TTV_OminusC, yerr=sim_TTV_errors, ecolor='k', zorder=1, fmt='none')
				plt.plot(np.linspace(np.nanmin(sim_TTV_epochs), np.nanmax(sim_TTV_epochs), 100), np.linspace(0,0,100), color='k', linestyle='--', zorder=0)
				plt.plot(interp_epochs, sinecurve, c='r', linestyle=':', linewidth=2)
				plt.xlabel('Epoch')
				plt.ylabel('O - C [s]')
				plt.title(r'$P = $'+str(round(peak_power_period,2))+r' epochs, $N_{S} = $'+str(sim_nmoons)+r', $\Delta \mathrm{BIC} = $'+str(round(deltaBIC, 2)))
				plt.show()





			#### plot the positions
			nparticles = sim_xpos.shape[0]
			if show_individual_plots == 'y':
				for particle in np.arange(0,nparticles,1):
					part_xpos, part_ypos = sim_xpos[particle], sim_ypos[particle]
					plt.plot(part_xpos, part_ypos, alpha=0.7)
					plt.xlabel(r'$a / a_{I}$')
					plt.ylabel(r'$a / a_{I}$')
					plt.title("MEGNO = "+str(round(sim_MEGNO, 2))+', stability prob = '+str(round(sim_SPOCKprob*100,2))+'%')
				plt.show()


			#### WRITE OUT THIS INFORMATION TO THE SIMULATION OBSERVATIONS SUMMARY FILE !!!! THESE CAN BEN ANN INPUTS.
			if plot_individual == 'n':
				sim_obs_summary = open(projectdir+'/simulated_observations.csv', mode='a')
				#sim_obs_summary.write('sim,Pplan_days,ntransits,TTV_rmsamp_sec,TTVperiod_epochs,peak_power,fit_sineamp,deltaBIC\n')
				sim_obs_summary.write(str(sim)+','+str(sim_Pplan_days)+','+str(sim_ntransits)+','+str(sim_TTV_rmsamp)+','+str(sim_TTV_epochs)+','+str(peak_power)+','+str(popt[0])+','+str(deltaBIC)+'\n')
				sim_obs_summary.close()	

		except:
			continue 




	#### plot Delta-BIC versus MEGNO and SPOCK survival probability
	deltaBIC_list = np.array(deltaBIC_list)
	peak_power_periods_list = np.array(peak_power_periods_list)


	stable_megno_idxs = np.where((megno_vals >= 1.97) & (megno_vals <= 2.18))[0]
	stable_spockprobs = np.where(spockprobs >= 0.9)[0]
	unstable_megno_idxs = np.concatenate((np.where(megno_vals < 1.97)[0], np.where(megno_vals > 2.18)[0]))
	unstable_spockprobs = np.where(spockprobs < 0.9)[0]







	#### plot the inferred periods against the BIC values
	plt.scatter(peak_power_periods_list[stable_megno_idxs], deltaBIC_list[stable_megno_idxs], facecolor='DodgerBlue', edgecolor='k', alpha=0.7, s=20)
	plt.xlabel('TTV period [epochs]')
	plt.ylabel(r'$\Delta \mathrm{BIC}$')
	plt.show()


	##### DO THE SAME WITH THE TTV RMS AMPLITUDES
	plt.scatter(np.array(summaryfile['TTV_rmsamp_sec'])[stable_megno_idxs]/60, deltaBIC_list[stable_megno_idxs], facecolor='DodgerBlue', edgecolor='k', alpha=0.7, s=20)
	plt.xlabel('TTV r.m.s. [minutes]')
	plt.ylabel(r'$\Delta \mathrm{BIC}$')
	plt.show()	


	##### TTV RMS vs Planet Period
	Pplans = np.array(summaryfile['Pplan_days']).astype(float)
	plt.scatter(Pplans[stable_megno_idxs], np.array(summaryfile['TTV_rmsamp_sec'])[stable_megno_idxs]/60, facecolor='DodgerBlue', edgecolor='k', alpha=0.7, s=20)
	plt.xlabel(r'$P_P$ [days]')
	plt.ylabel('TTV r.m.s. [minutes]')
	plt.show()	




	#### break it up by number of moons!
	for moon_number in np.arange(1,6,1):
		nmoons_idxs = np.where(nmoons == moon_number)[0]
		nmoons_stable_idxs = np.intersect1d(nmoons_idxs, stable_megno_idxs)
		plt.scatter(peak_power_periods_list[nmoons_stable_idxs], deltaBIC_list[nmoons_stable_idxs], edgecolor='k', alpha=0.5, s=20, label=str(moon_number))
	plt.plot(np.linspace(2,500,1000), np.linspace(0,0,1000), c='k', linestyle='--')
	plt.xlabel('TTV period [epochs]')
	plt.ylabel(r'$\Delta \mathrm{BIC}$')
	plt.xscale('log')
	plt.legend()
	plt.tight_layout()
	plt.show()



	fig, ax = plt.subplots(5, sharex=True, figsize=(6,10))
	histdict = {}
	for moon_number in np.arange(1,6,1):
		nmoon_idxs = np.where(nmoons == moon_number)[0]
		good_BIC_idxs = np.where(deltaBIC_list < -2)[0] #### positive evidence for a moon
		nmoons_stable_idxs = np.intersect1d(nmoon_idxs, stable_megno_idxs)
		nmoons_stable_good_BIC_idxs = np.intersect1d(nmoons_stable_idxs, good_BIC_idxs)
		TTV_period_bins = np.arange(2,20,1)

		print('moon number = ', moon_number)
		print('# of systems = ', len(nmoon_idxs))
		print('# of these that are stable = ', len(nmoons_stable_idxs))
		print('# stable with good BIC = ', len(nmoons_stable_good_BIC_idxs))
		print(" ")
		
		histdict['hist'+str(moon_number)] = ax[moon_number-1].hist(peak_power_periods_list[nmoons_stable_good_BIC_idxs], bins=TTV_period_bins, facecolor='DodgerBlue', edgecolor='k', alpha=0.7)
		ax[moon_number-1].set_ylabel(r'$N = $'+str(moon_number))

	ax[4].set_xlabel('TTV period [epochs]')
	plt.tight_layout()
	plt.show()



	##### LOOK AT TTV PERIODS AS A FUNCTION OF PLANETARY PERIOD!

	#### FIT THE UPPER LIMIT FOR THIS TRIANGLE!!!!

	Pplan_bins = np.arange(10,1505,5)
	#Pplan_bins = np.logspace(np.log10(10), np.log10(1505), 100)
	Pplan_bin_max_TTVs = []
	for nPPb, PPb in enumerate(Pplan_bins):
		try:
			bin_idxs = np.where((Pplans >= Pplan_bins[nPPb]) & (Pplans < Pplan_bins[nPPb+1]))[0]
		except:
			Pplan_bin_max_TTVs.append(np.nan)
			break

		#### need to find the intersections with the good_BIC_idxs
		bin_idxs = np.intersect1d(bin_idxs, good_BIC_idxs)
		bin_idxs = np.intersect1d(bin_idxs, stable_megno_idxs)

		bin_TTV_periods = peak_power_periods_list[bin_idxs]
		try:
			highest_TTV_in_bin = np.nanmax(bin_TTV_periods)
		except:
			highest_TTV_in_bin = np.nan
		Pplan_bin_max_TTVs.append(highest_TTV_in_bin)

	Pplan_bin_max_TTVs = np.array(Pplan_bin_max_TTVs)

	Pplan_bins = Pplan_bins[np.isfinite(Pplan_bin_max_TTVs)]
	Pplan_bin_max_TTVs = Pplan_bin_max_TTVs[np.isfinite(Pplan_bin_max_TTVs)]


	#### break it up by number of moons!
	for moon_number in np.arange(1,6,1):
		nmoon_idxs = np.where(nmoons == moon_number)[0]
		good_BIC_idxs = np.where(deltaBIC_list < -2)[0] #### positive evidence for a moon
		nmoons_stable_idxs = np.intersect1d(nmoon_idxs, stable_megno_idxs)
		nmoons_stable_good_BIC_idxs = np.intersect1d(nmoons_stable_idxs, good_BIC_idxs)
		plt.scatter(np.array(summaryfile['Pplan_days'])[nmoons_stable_good_BIC_idxs], peak_power_periods_list[nmoons_stable_good_BIC_idxs], edgecolor='k', alpha=0.2, s=20, label=str(moon_number))
	
	plt.scatter(Pplan_bins+2.5, Pplan_bin_max_TTVs, c='k', alpha=0.5, marker='+', s=20)

	#### fit a power law to this
	def powerlaw(xvals, amplitude, exponent):
		return amplitude * xvals**exponent

	pl_popt, pl_pcov = curve_fit(powerlaw, Pplan_bins, Pplan_bin_max_TTVs)

	plt.plot(Pplan_bins, powerlaw(Pplan_bins, *pl_popt), c='k', linestyle='--', linewidth=2)

	#plt.plot(np.linspace(2,500,1000), np.linspace(0,0,1000), c='k', linestyle='--')
	plt.xlabel('Planet Period')
	plt.ylabel('TTV period [epochs]')
	plt.xscale('log')
	plt.yscale('log')
	plt.legend()
	plt.tight_layout()
	plt.show()








	#### SCATTER PLOTS
	plt.scatter(megno_vals, deltaBIC_list, facecolor='DodgerBlue', edgecolor='k', alpha=0.7)
	plt.plot(np.linspace(np.nanmin(megno_vals), np.nanmax(megno_vals), 100), np.linspace(0,0,100), c='k', linestyle='--')
	plt.xlabel('MEGNO')
	plt.ylabel(r'$\Delta \mathrm{BIC}$')
	plt.xscale('log')
	plt.show()

	plt.scatter(spockprobs*100, deltaBIC_list, facecolor='DodgerBlue', edgecolor='k', alpha=0.7)
	plt.xlabel('survival probability [%]')
	plt.ylabel(r'$\Delta \mathrm{BIC}$')
	plt.show()

	###### HISTOGRAMS
	######### THE STABLE CASE!!!!!
	fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(6,10))
	n1,b1,e1 = ax1.hist(deltaBIC_list[stable_megno_idxs], bins=np.arange(-10,11,1), facecolor='NavajoWhite', edgecolor='k', alpha=1, zorder=1)
	ax1.plot(np.linspace(np.nanmedian(deltaBIC_list[stable_megno_idxs]),np.nanmedian(deltaBIC_list[stable_megno_idxs]),100), np.linspace(0,1.1*np.nanmax(n1),100), c='k', linestyle='--')
	ax1.axvspan(-11, 0, alpha=0.2, color='green', zorder=0)
	ax1.axvspan(0, 11, alpha=0.2, color='red', zorder=0)
	ax1.set_ylabel(r'$1.97\leq \mathrm{MEGNO} \leq 2.18$')
	ax1.set_xlim(-10,10)
	ax1.set_ylim(0,1.1*np.nanmax(n1))
	n2,b2,e2 = ax2.hist(deltaBIC_list[stable_spockprobs], bins=np.arange(-10,11,1), facecolor='NavajoWhite', edgecolor='k', alpha=1, zorder=1)
	ax2.plot(np.linspace(np.nanmedian(deltaBIC_list[stable_spockprobs]),np.nanmedian(deltaBIC_list[stable_spockprobs]),100), np.linspace(0,1.1*np.nanmax(n2),100), c='k', linestyle='--')
	ax2.axvspan(-11, 0, alpha=0.2, color='green', zorder=0)
	ax2.axvspan(0, 11, alpha=0.2, color='red', zorder=0)
	ax2.set_ylabel(r'SPOCK $P_{\mathrm{stable}} \geq 0.9$')
	ax2.set_xlim(-10,10)
	ax2.set_ylim(0,1.1*np.nanmax(n2))
	
	ax2.set_xlabel(r'$\Delta \mathrm{BIC}$')
	ax1.set_title('stable systems')
	plt.tight_layout()
	plt.show()


	###### HISTOGRAMS
	fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(6,10))
	n1,b1,e1 = ax1.hist(deltaBIC_list[unstable_megno_idxs], bins=np.arange(-10,11,1), facecolor='NavajoWhite', edgecolor='k', alpha=1, zorder=1)
	ax1.plot(np.linspace(np.nanmedian(deltaBIC_list[unstable_megno_idxs]),np.nanmedian(deltaBIC_list[unstable_megno_idxs]),100), np.linspace(0,1.1*np.nanmax(n1),100), c='k', linestyle='--')
	ax1.axvspan(-11, 0, alpha=0.2, color='green', zorder=0)
	ax1.axvspan(0, 11, alpha=0.2, color='red', zorder=0)
	ax1.set_ylabel(r'$\mathrm{MEGNO} < 1.97 \, \mathrm{;} > 2.18$')
	ax1.set_xlim(-10,10)
	ax1.set_ylim(0,1.1*np.nanmax(n1))
	n2,b2,e2 = ax2.hist(deltaBIC_list[unstable_spockprobs], bins=np.arange(-10,11,1), facecolor='NavajoWhite', edgecolor='k', alpha=1, zorder=1)
	ax2.plot(np.linspace(np.nanmedian(deltaBIC_list[unstable_spockprobs]),np.nanmedian(deltaBIC_list[unstable_spockprobs]),100), np.linspace(0,1.1*np.nanmax(n2),100), c='k', linestyle='--')
	ax2.axvspan(-11, 0, alpha=0.2, color='green', zorder=0)
	ax2.axvspan(0, 11, alpha=0.2, color='red', zorder=0)
	ax2.set_ylabel(r'SPOCK $P_{\mathrm{stable}} < 0.9$')
	ax2.set_xlim(-10,10)
	ax2.set_ylim(0,1.1*np.nanmax(n2))
	
	ax2.set_xlabel(r'$\Delta \mathrm{BIC}$')
	ax1.set_title('unstable systems')
	plt.tight_layout()
	plt.show()





except:
	traceback.print_exc()
	raise Exception('something happened.')

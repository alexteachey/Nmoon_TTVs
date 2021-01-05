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
		fit_TTVs = input('Do you want to fit the TTVs (slow)? y/n: ')
		sim_obs_summary = open(projectdir+'/simulated_observations.csv', mode='w')
		sim_obs_summary.write('sim,Nmoons,Pplan_days,ntransits,TTV_rmsamp_sec,TTVperiod_epochs,peak_power,fit_sineamp,deltaBIC,MEGNO,SPOCK_prob\n')
		sim_obs_summary.close()



	#### NEW SUMMARY STATISTICS LISTS
	deltaBIC_list = []
	max_fractional_delta_rvals = []
	Msats_over_Mps = []
	peak_power_periods_list = []
	nmoons = np.array(summaryfile['nmoons'])
	megno_vals = np.array(summaryfile['MEGNO'])
	spockprobs = np.array(summaryfile['SPOCK_survprop'])


	first_n1, first_n2, first_n3, first_n4, first_n5 = 'n', 'n', 'n', 'n', 'n'
	highest_number = 1
	for nsim, sim in enumerate(sims):

		print("sim # =", nsim)
		try:
			#### MODEL INPUTS
			#dicttime1 = time.time()
			sim_model_dict = pickle.load(open(modeldictdir+'/TTVsim'+str(sim)+'_system_dictionary.pkl', "rb"))

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

			if (sim_nmoons <= 2) and (sim_MEGNO > 1.98) and (sim_MEGNO < 2.16):
				stable = 'y'
			elif (sim_nmoons > 2) and (sim_SPOCKprob > 0.8):
				stable = 'y'
			else:
				stable = 'n'

			if stable == 'n':
				print('UNSTABLE!')
				continue



			if (sim_nmoons == 1) and (first_n1 == 'n') and (highest_number == sim_nmoons):
				first_example = 'y'
				first_n1 = 'y'
				highest_number += 1
			elif (sim_nmoons == 2) and (first_n2 == 'n') and (highest_number == sim_nmoons):
				first_example = 'y'
				first_n2 = 'y'
				highest_number += 1
			elif (sim_nmoons == 3) and (first_n3 == 'n')and (highest_number == sim_nmoons):
				first_example = 'y'
				first_n3 = 'y'
				highest_number += 1
			elif (sim_nmoons == 4) and (first_n4 == 'n') and (highest_number == sim_nmoons):
				first_example = 'y'
				first_n4 = 'y'
				highest_number += 1
			elif (sim_nmoons == 5) and (first_n5 == 'n') and (highest_number == sim_nmoons):
				first_example = 'y'
				first_n5 = 'y'
				highest_number += 1
			else:
				first_example = 'n'

			if sim_nmoons == 1:
				continue #### we don't need to plot them here


			sim_masses, sim_mass_ratios, sim_smas, sim_smas_fracHill = [], [], [], []
			moon_labels = ['I', 'II', 'III', 'IV', 'V']
			sim_RHill = sim_model_dict['Planet']['RHill']
			sim_mplan = sim_model_dict['Planet']['m']
			for moon in np.arange(0,sim_nmoons,1):
				moon_a, moon_m = sim_model_dict[moon_labels[moon]]['a'], sim_model_dict[moon_labels[moon]]['m']
				sat_mass_ratio = moon_m / sim_mplan
				sim_masses.append(moon_m)
				sim_mass_ratios.append(sat_mass_ratio)
				sim_smas.append(moon_a)
				sim_smas_fracHill.append(moon_a / sim_RHill)
			sim_masses, sim_mass_ratios, sim_smas, sim_smas_fracHill = np.array(sim_masses), np.array(sim_mass_ratios), np.array(sim_smas), np.array(sim_smas_fracHill)
			if np.any(sim_smas_fracHill) > 0.4895:
				print("Moon beyond 0.4895 RHill")
				time.sleep(1)


			#plt.scatter(sim_smas_fracHill, np.linspace(nsim,nsim,sim_nmoons), s=np.log10(1/sim_mass_ratios)**2, edgecolor='k', facecolor='DodgerBlue', alpha=0.7)

			colors = cm.inferno(np.linspace(0,1,5))
			#plt.plot(sim_smas_fracHill, sim_mass_ratios, color=colors[sim_nmoons-1], linewidth=2, zorder=0)
			#### normalize instead to the mass of satellite 1!
			if first_example == 'y':
				plt.plot(sim_smas_fracHill, sim_masses / sim_masses[0], color=colors[sim_nmoons-1], linewidth=2, alpha=0.5, label='N = '+str(sim_nmoons))
			else:			
				plt.plot(sim_smas_fracHill, sim_masses / sim_masses[0], color=colors[sim_nmoons-1], linewidth=2, alpha=0.5)
		except:
			traceback.print_exc()
			continue


	plt.xlabel(r'$a / R_{\mathrm{Hill}}$')
	#plt.ylabel(r'$M_S / M_P$')
	plt.ylabel(r'$M_S / M_{S_1}$')
	plt.yscale('log')
	plt.legend()
	plt.show()






















	for sim in sims:
		try:
			print('sim # '+str(sim))
			#### MODEL INPUTS
			#dicttime1 = time.time()
			sim_model_dict = pickle.load(open(modeldictdir+'/TTVsim'+str(sim)+'_system_dictionary.pkl', "rb"))
			#dicttime2 = time.time()
			#print('dictionary load time = ', dicttime2 - dicttime1)		

			#### TTV file
			#ttvtime1 = time.time()
			sim_TTVs = pandas.read_csv(ttvfiledir+'/TTVsim'+str(sim)+'_TTVs.csv')
			sim_TTV_epochs = np.array(sim_TTVs['epoch']).astype(int)
			sim_TTV_OminusC = np.array(sim_TTVs['TTVob']).astype(float)
			sim_TTV_errors = np.array(sim_TTVs['timing_error']).astype(float)
			#ttvtime2 = time.time()
			#print('ttv load time =', ttvtime2 - ttvtime1)

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
				plt.plot(periods, powers, c='DodgerBlue', linewidth=2, alpha=0.7)
				plt.xlabel('Period [epochs]')
				plt.ylabel('Power')
				plt.xscale('log')
				plt.title('best period = '+str(round(peak_power_period, 3))+' epochs')
				plt.show()




			#### perform a curve_fit to find amplitude and offset, to be plotted below (and for Delta-BIC -- or just delta X^2)?
			if fit_TTVs == 'y':
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
				#deltaBIC_list.append(deltaBIC) #### MOVED TO THE END -- SO THAT YOUR INDEXING IS OK!


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
			maximum_fractional_delta_r = np.nanmax(fractional_delta_rvals)
			#max_fractional_delta_rvals.append(maximum_fractional_delta_r)  ### MOVED TO THE END, TO KEEP LISTS EVEN



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
				#sim_obs_summary.write('sim,Pplan_days,ntransits,	TTV_rmsamp_sec,TTVperiod_epochs,peak_power,fit_sineamp,deltaBIC,MEGNO,SPOCK_prob\n')
				sim_obs_summary.write(str(sim)+','+str(sim_nmoons)+','+str(sim_Pplan_days)+','+str(sim_ntransits)+','+str(sim_TTV_rmsamp)+','+str(sim_TTVperiod_epochs)+','+str(peak_power)+','+str(popt[0])+','+str(deltaBIC)+','+str(sim_MEGNO)+','+str(sim_SPOCKprob)+'\n')
				sim_obs_summary.close()


			#### PLACE MISCELANEOUS LIST APPENDS DOWN HERE.
			Msats_over_Mps.append(sim_Msats_over_Mp)
			deltaBIC_list.append(deltaBIC) #### MOVED TO THE END -- SO THAT YOUR INDEXING IS OK!
			max_fractional_delta_rvals.append(maximum_fractional_delta_r)  ### MOVED TO THE END, TO KEEP LISTS EVEN
			peak_power_periods_list.append(peak_power_period)

		except:
			#### APPENDING NaNs so we can keep all the lists the same length!
			Msats_over_Mps.append(np.nan)
			deltaBIC_list.append(np.nan)
			max_fractional_delta_rvals.append(np.nan) 
			peak_power_periods_list.append(np.nan)


			continue 



	#### MAKE THE LISTS INTO ARRAYS
	deltaBIC_list = np.array(deltaBIC_list)
	peak_power_periods_list = np.array(peak_power_periods_list)
	max_fractional_delta_rvals = np.array(max_fractional_delta_rvals)
	Msats_over_Mps = np.array(Msats_over_Mps)


	##### DATA CUTS
	stable_megno_idxs = np.where((megno_vals >= 1.97) & (megno_vals <= 2.18))[0]
	stable_spockprobs = np.where(spockprobs >= 0.8)[0] #### UPDATE BASED ON DEARTH OF P > 0.9 systems in the new paradigm.
	unstable_megno_idxs = np.concatenate((np.where(megno_vals < 1.97)[0], np.where(megno_vals > 2.18)[0]))
	unstable_spockprobs = np.where(spockprobs < 0.9)[0]
	stable_idxs = []

	#### STABILITY CHECK
	for idx in np.arange(0,len(megno_vals),1):
		if (idx in stable_spockprobs):
			stable_idxs.append(idx) #### if SPOCK probability is good, we go with this
		
		elif (np.isfinite(spockprobs[idx]) == False) and (idx in stable_megno_idxs): ### if SPOCK prob is unavailable but the MEGNO is good, we go with this
			stable_idxs.append(idx) 
		
		elif (spockprobs[idx] < 0.9) and (idx in stable_megno_idxs):
			continue 
	stable_idxs = np.array(stable_idxs)		



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



	#### fit exponential curves to these histogram points
	def TTV_curve(xvals, amplitude, beta):
		#### returns an function of the form amplitude * np.exp(-xvals / beta)
		return amplitude * np.exp(-xvals / beta)


	fig, ax = plt.subplots(5, sharex=True, figsize=(6,10))
	histdict = {}
	for moon_number in np.arange(1,6,1):
		nmoon_idxs = np.where(nmoons == moon_number)[0]
		good_BIC_idxs = np.where(deltaBIC_list < -2)[0] #### positive evidence for a moon
		nmoons_stable_idxs = np.intersect1d(nmoon_idxs, stable_idxs)
		nmoons_stable_good_BIC_idxs = np.intersect1d(nmoons_stable_idxs, good_BIC_idxs)
		TTV_period_bins = np.arange(2,20,1)


		print('moon number = ', moon_number)
		print('# of systems = ', len(nmoon_idxs))
		print('# of these that are stable = ', len(nmoons_stable_idxs))
		print('# stable with good BIC = ', len(nmoons_stable_good_BIC_idxs))
		print(" ")
		
		histdict['hist'+str(moon_number)] = ax[moon_number-1].hist(peak_power_periods_list[nmoons_stable_good_BIC_idxs], bins=TTV_period_bins, facecolor='DodgerBlue', edgecolor='k', alpha=0.7)
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
		nmoons_stable_idxs = np.intersect1d(nmoon_idxs, stable_idxs)
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


	#### GENERATE THE SAME PLOT, BUT WITH A GAUSSIAN KERNEL DENSITY ESTIMATOR (COMPARE TO RESULTS FROM "real_planet_TTV_analyzer.py")

	good_BIC_stable_idxs = np.intersect1d(good_BIC_idxs, stable_idxs)
	P_plans = np.array(summaryfile['Pplan_days'])[good_BIC_stable_idxs]
	P_TTVs = peak_power_periods_list[good_BIC_stable_idxs]
	kdestack = np.vstack((P_plans, P_TTVs))

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
	xbins = np.logspace(np.log10(10), np.log10(1500), 20)
	ybins = np.logspace(np.log10(2), np.log10(100), 20)
	TTV_Pplan_hist2d = np.histogram2d(P_plans, P_TTVs, bins=[xbins, ybins])
	plt.imshow(TTV_Pplan_hist2d[0].T, origin='lower', cmap=cm.coolwarm)
	plt.xticks(ticks=np.arange(0,len(xbins),5), labels=np.around(np.log10(xbins[::5]),2))
	plt.yticks(ticks=np.arange(0,len(ybins),5), labels=np.around(np.log10(ybins[::5]), 2))
	plt.xlabel(r'$\log_{10} \, P_{\mathrm{P}}$ [days]')
	plt.ylabel(r'$\log_{10} \, P_{\mathrm{TTV}}$ [epochs]')
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
	######### THE STABLE CASE!!!!! -- COMPARING DISTRIBUTIONS FOR MEGNO AND SPOCK PROBABILITIES
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
	######### THE STABLE CASE!!!!! (COMBINED MEGNO / SPOCK STABLE_IDXs)
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




	###### HISTOGRAMS --- COMPARING *UNSTABLE* MEGNO AND SPOCK DISTRIBUTIONS.
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

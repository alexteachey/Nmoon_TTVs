from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas
import os
import traceback
import pickle

plt.rcParams["font.family"] = 'serif'

#### THIS SCRIPT WILL GENERATE PERIOD-RATIO HISTOGRAMS OF THE MOONS IN YOUR SIMULATIONS
try:
	sim_prefixes = ['RUN4', '', 'MMR2']
	#sim_prefix = input('What is the sim_prefix? ')


	sim_prefix_all_period_ratios_dict = {}


	for sim_prefix in sim_prefixes:


		projectdir = '/run/media/amteachey/Auddy_Akiti/Teachey/Nmoon_TTVs'
		if len(sim_prefix) == 0:
			modeldir = projectdir+'/sim_model_settings'
			simsum = pandas.read_csv(projectdir+'/simulation_summary.csv')
		else:
			modeldir = projectdir+'/'+sim_prefix+'_sim_model_settings'
			simsum = pandas.read_csv(projectdir+'/'+sim_prefix+'_simulation_summary.csv')

		simnumber = np.array(simsum['sim']).astype(int)
		nmoons = np.array(simsum['nmoons']).astype(int)
		megnos = np.array(simsum['MEGNO']).astype(float)
		spocks = np.array(simsum['SPOCK_survprop']).astype(float)


		all_period_ratios = []

		nsims = len(os.listdir(modeldir))

		for sim in np.arange(1,nsims+1,1):

			sim_idx = int(np.where(int(sim) == simnumber)[0])
			sim_nmoons = nmoons[sim_idx]
			sim_megno = megnos[sim_idx]
			sim_spock = spocks[sim_idx]

			#### CHECK STABILITY
			if (sim_nmoons < 3) and ((sim_megno > 1.96) and (sim_megno < 2.22)):
				stable = 'y'
			elif (sim_nmoons < 3) and ((sim_megno < 1.96) or (sim_megno > 2.22)):
				stable = 'n'
			elif (sim_nmoons >= 3) and (sim_spock >= 0.9):
				stable = 'y'
			elif (sim_nmoons >= 3) and (sim_spock < 0.9):
				stable = 'n'

			if stable == 'n':
				print('unstable! continue...')
				continue

			print('sim # '+str(sim)+' of '+str(nsims))
			simdict = pickle.load(open(modeldir+'/TTVsim'+str(sim)+'_system_dictionary.pkl', 'rb'))
			simdict_keys = simdict.keys()
			sim_nmoons = len(simdict_keys)-1 #### planet, and all the moons
			moon_labels = ['I', 'II', 'III', 'IV', 'V']
			sim_period_ratios = []
			sim_moon_keys = moon_labels[:sim_nmoons] ### if sim_nmoons == 3, you have moon_labels[:3] = ['I', 'II', 'III']
			num_to_subtract = 1
			for i in np.arange(1,sim_nmoons,1): #if sim_nmoons == 3, this is np.array([1,2])		
				#### do all pairs, not just adjacent planets!		
				try:	
					outer_moon = simdict[sim_moon_keys[i]]
					inner_moon = simdict[sim_moon_keys[i-num_to_subtract]]
					outer_moon_period_seconds = outer_moon['P']
					inner_moon_period_seconds = inner_moon['P']
					period_ratio = outer_moon_period_seconds / inner_moon_period_seconds
					all_period_ratios.append(period_ratio)
					num_to_subtract += 1
				except:
					continue

		all_period_ratios = np.array(all_period_ratios)
		sim_prefix_all_period_ratios_dict[sim_prefix] = all_period_ratios





	sim_prefix_labels = {'RUN4':'fixed host', '':'variable host', 'MMR2':'resonant chain'}

	fig, ax = plt.subplots(len(sim_prefixes), sharex=True, figsize=(6,8))
	for nsim, sim_prefix in enumerate(sim_prefixes):
		all_period_ratios = sim_prefix_all_period_ratios_dict[sim_prefix]

		period_ratio_bins = np.arange(0.95, 4.45, 0.1)
		n, bins, edges = ax[nsim].hist(all_period_ratios, bins=period_ratio_bins, facecolor='DodgerBlue', edgecolor='k', alpha=0.7)
		ax[nsim].set_ylabel(sim_prefix_labels[sim_prefix])
		#### plot the common period ratios
		ax[nsim].plot(np.linspace(2, 2, 100), np.linspace(0,1.2*np.nanmax(n), 100), c='k', linestyle='--', alpha=0.5)
		ax[nsim].plot(np.linspace(3/2, 3/2, 100), np.linspace(0,1.2*np.nanmax(n), 100), c='k', linestyle='--', alpha=0.5, label='3:2')
		ax[nsim].plot(np.linspace(4/3, 4/3, 100), np.linspace(0,1.2*np.nanmax(n), 100), c='k', linestyle='--', alpha=0.5, label='4:3')
		ax[nsim].plot(np.linspace(5/4, 5/4, 100), np.linspace(0,1.2*np.nanmax(n), 100), c='k', linestyle='--', alpha=0.5, label='5:4')
		ax[nsim].plot(np.linspace(5/2, 5/2, 100), np.linspace(0,1.2*np.nanmax(n), 100), c='k', linestyle='--', alpha=0.5, label='5:2')
		ax[nsim].plot(np.linspace(7/2, 7/2, 100), np.linspace(0,1.2*np.nanmax(n), 100), c='k', linestyle='--', alpha=0.5, label='7:2')
		ax[nsim].plot(np.linspace(3, 3, 100), np.linspace(0, 1.2*np.nanmax(n), 100), c='k', linestyle='--', alpha=0.5, label='3:1')
		ax[nsim].plot(np.linspace(4, 4, 100), np.linspace(0, 1.2*np.nanmax(n), 100), c='k', linestyle='--', alpha=0.5, label='4:1')
		#ax[nsim].plot(np.linspace(3,3,100), np.linspace(0.1.05*np.nanmax(n),100),c='k', linestyle=)
		#plt.plot(np.linspace(6/5, 6/5, 100), np.linspace(0,1.05*np.nanmax(n), 100), c='k', linestyle='--')
		#plt.plot(np.linspace(7/6, 7/6, 100), np.linspace(0,1.05*np.nanmax(n), 100), c='k', linestyle='--')
		ax[nsim].set_ylim(0, 1.2*np.nanmax(n))


	ax[nsim].set_xlabel(r'$P_{\mathrm{outer}} \, / \, P_{\mathrm{inner}}$')
	plt.subplots_adjust(left=0.125, bottom=0.09, right=0.9, top=0.95, wspace=0.2, hspace=0.2)
	plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/period_ratio_stable_3histograms.png', dpi=300)
	plt.savefig('/data/tethys/Documents/Projects/NMoon_TTVs/Plots/period_ratio_stable_3histograms.pdf', dpi=300)
	plt.show()

except:
	traceback.print_exc()




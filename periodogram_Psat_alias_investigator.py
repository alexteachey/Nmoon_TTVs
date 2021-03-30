from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas
import os
import traceback
from astropy.timeseries import LombScargle
import matplotlib.cm as cm


#### this script will examine how the periodograms evolve as Psat -- > Pepoch (P_plan).
"""
That is, what is the effect of undersampling on the inferred period, or shape of the periodogram? 

Try this: look at the evolution of the periodogram as you change the period of the moon.


"""










try:


	ntrials = int(input('How many experiments you want to run? '))

	for trial in np.arange(0,ntrials,1):
		##### SEE HOW THE OBSERVATIONAL CADENCE PLAYS WITH THE SIGNAL
		nsignals = np.random.choice(np.arange(1,5,1))
		#amplitudes = np.random.random(size=nsignals)
		amplitudes = np.linspace(1,1,nsignals)
		periods = np.random.choice(np.arange(2,21,1), size=nsignals) + np.random.random(size=nsignals)
		phases = np.random.choice(np.linspace(-2*np.pi, 2*np.pi, 100), size=nsignals)

		periods_to_probe = np.linspace(2,40,1000)
		freqs_to_probe = 1 / periods_to_probe
		lc_time_array = np.arange(0,60,1) 
		hc_time_array = np.arange(0,60.01,0.01)
		hc_delta_funcs = np.zeros(shape=((len(hc_time_array),)))
		for nhcta, hcta in enumerate(hc_time_array):
			if hcta % int(hcta) == 0.0:
				hc_delta_funcs[nhcta] = 1.0

		signal = np.zeros(shape=(len(lc_time_array),))
		for amp,per,ph in zip(amplitudes, periods, phases):
			signal += amp*np.sin((2*np.pi*(1/per)*lc_time_array) + ph)

		signal_LS = LombScargle(lc_time_array, signal)
		signal_LS_powers = signal_LS.power(freqs_to_probe)
		cadence_LS = LombScargle(hc_time_array, hc_delta_funcs)
		cadence_LS_powers = cadence_LS.power(freqs_to_probe)
		signal_div_cadence_LS_powers = signal_LS_powers / cadence_LS_powers

		fig, ax = plt.subplots(4, figsize=(6,8))
		ax[0].plot(lc_time_array, signal, color='LightCoral', linewidth=2, alpha=0.7)
		ax[1].plot(periods_to_probe, signal_LS_powers, c='k', alpha=0.7)
		ax[2].plot(periods_to_probe, cadence_LS_powers, c='k', alpha=0.7)
		ax[3].plot(periods_to_probe, signal_div_cadence_LS_powers, c='k', alpha=0.7)
		ax[0].set_title('# signals = '+str(nsignals))
		plt.show()















	Pmoons = np.arange(2,21,1) #### days
	colors = cm.viridis(np.linspace(0,1,len(Pmoons)))

	baseline = 3650 ### days
	Pplans = np.arange(50,1010,10)
	ntransits = baseline // Pplans 
	periods_to_probe = np.logspace(np.log10(2),np.log10(np.nanmax(ntransits)),1000)
	freqs_to_probe = 1 / periods_to_probe

	master_GT_Psats_over_Ps = []
	master_peak_powers = []
	master_LS_Psats_over_Ps = []


	for nt,Pp in zip(ntransits, Pplans):
		print('Pplan = ', Pp)
		time_array = np.arange(0,nt,1) #### each epoch is 1
		time_array = np.sort(np.random.choice(time_array, size=int(0.6*len(time_array))))

		Psats_over_Pp = Pmoons / Pp #### ratio of moon period to planet period (must be < 1!) 
		LS_Psats_over_Pp = []
		fsats_over_fp = Pp / Pmoons #### equivalent to 1 / Psats_over_Pp 
		##### since in this context Pp = 1, the fraction above Psat_over_Pp is the period of the moon.
		#sat_sines = np.sin(2*np.pi*fsats_over_fp*time_array + np.pi)
		sat_sines = np.zeros(shape=(len(Pmoons), len(time_array)))
		for nfsofp, fsofp in enumerate(fsats_over_fp):
			sat_sines[nfsofp] = 5 * np.sin((2*np.pi*fsofp*time_array) + np.pi)

			#### noise it up!
			sat_sines[nfsofp] = np.random.normal(loc=sat_sines[nfsofp], scale=0.2*np.nanmax(sat_sines[nfsofp]))

			#plt.scatter(time_array, sat_sines[nfsofp], facecolor='LightCoral', edgecolor='k', s=20, alpha=0.7)
			#plt.plot(time_array, sat_sines[nfsofp], c='k', alpha=0.5, linestyle=':')
			#plt.show()

		#for nss, ss in enumerate(sat_sines):
		for nss in np.arange(0,sat_sines.shape[0],1):
			ss = sat_sines[nss]
			#### run the periodogram
			ss_LS = LombScargle(time_array, ss)
			ss_LS_powers = ss_LS.power(freqs_to_probe)

			##### TRY TO DIVIDE OUT THE OBSERVING CADENCE PERIODOGRAM!
			hc_time_array = np.arange(0,nt,0.01)
			hc_delta_funcs = np.zeros(shape=(len(hc_time_array),))
			for nhcta, hcta in enumerate(hc_time_array):
				if hcta % int(hcta) == 0.0:
					hc_delta_funcs[nhcta] = 1.0

			LS_obs_cadence = LombScargle(hc_time_array, hc_delta_funcs)
			LS_obs_cadence_powers = LS_obs_cadence.power(freqs_to_probe)

			fig, ax = plt.subplots(3, sharex=True)
			ax[0].plot(periods_to_probe, ss_LS_powers)
			ax[0].set_ylabel('signal')
			ax[1].plot(periods_to_probe, LS_obs_cadence_powers)
			ax[1].set_ylabel('cadence')
			ax[2].plot(periods_to_probe, ss_LS_powers / LS_obs_cadence_powers)
			ax[2].set_ylabel('signal / cadence')
			plt.xscale('log')
			plt.show()



			master_GT_Psats_over_Ps.append(Psats_over_Pp[nss]) ### GROUND TRUTH!
			master_peak_powers.append(np.nanmax(ss_LS_powers))
			master_LS_Psats_over_Ps.append(periods_to_probe[np.argmax(ss_LS_powers)]) ##### number epochs == LS Psats / Pp.
			LS_Psats_over_Pp.append(periods_to_probe[np.argmax(ss_LS_powers)])

			#plt.plot(periods_to_probe, ss_LS_powers, color=colors[nss], alpha=0.5)

		#plt.xscale('log')	
		#plt.title('Pp = '+str(Pp))
		#plt.show()

		plt.scatter(Psats_over_Pp, LS_Psats_over_Pp, c='LightCoral', edgecolor='k', s=20, alpha=0.7)
		plt.xlabel(r'$P_S / P_P$ (ground truth)')
		plt.ylabel(r'$P_S / P_P$ (Lomb-Scargle)')
		plt.show()



	plt.scatter(master_GT_Psats_over_Ps, master_LS_Psats_over_Ps, facecolor='LightCoral', edgecolor='k', s=20, alpha=0.5)
	plt.xlabel(r'$P_S / P_P$ (ground truth)')
	plt.ylabel(r'$P_S / P_P$ (Lomb-Scargle)')
	plt.show()

	good_idxs = np.where(np.array(master_LS_Psats_over_Ps) != 2.0)[0]

	n, bins, edges = plt.hist(np.array(master_LS_Psats_over_Ps)[good_idxs], bins=np.arange(2,100,2), facecolor='DodgerBlue', edgecolor='k', alpha=0.7)
	plt.xlabel('Peak power [epochs]')
	plt.show()





except:
	traceback.print_exc()

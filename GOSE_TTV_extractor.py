from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas
import os
import time
import traceback


#### this script will use the GOSE posteriors to produce a giant list (a la Holczer 2016) of transit timings.
"""
For each planet, do *joint posterior* draws for each segment, grab those timings, and use the median value across that segment.
Repeat for each segment.
The, fit a line to the timings -- that's your linear ephemeris.
Then you can compute O - C values from this.
CHECK YOUR LINEAR EPHEMERIS CALCULATION AGAINST THE FIDUCIAL VALUE.
Question: should you fit lines to each segment individually? Or one line to all of them? (I think the lines should be fit individually, but you should test this -- visualize it)
"""

try:

	GOSE_TTV_summaryfile = open('/data/tethys/Documents/Central_Data/GOSE_TTV_summaryfile.csv', mode='w')
	GOSE_TTV_summaryfile.write('KOI,Pplan_pub,Pplan_fit,epoch,tau_linear_ephemeris,tau_fit,OC_min,OCmin_err\n')
	GOSE_TTV_summaryfile.close()

	show_debug_plots = input("Do you want to show plots for debugging? y/n: ")

	TTVpostdir = '/run/media/amteachey/Auddy_Akiti/Teachey/Nmoon_TTVs/GOSE_TTVs/TTV_posteriors-master'
	TTVpostfiles = np.array(os.listdir(TTVpostdir))
	TTVpostfiles_copy = []
	for TTVpostfile in TTVpostfiles:
		if '.csv' in TTVpostfile:
			TTVpostfiles_copy.append(TTVpostfile)
	TTVpostfiles = np.array(TTVpostfiles_copy)


	### sort them by KOI
	KOI_nums = []
	for TTVpostfile in TTVpostfiles:
		KOI_nums.append(TTVpostfile[4:TTVpostfile.find('_')])
	KOI_nums = np.array(KOI_nums).astype(float)
	KOI_nums_argsort = np.argsort(KOI_nums)

	TTVpostfiles = TTVpostfiles[KOI_nums_argsort] 



	ndraws = 500

	for nTTVpostfile, TTVpostfile in enumerate(TTVpostfiles):
		if '.csv' not in TTVpostfile:
			continue

		KOI = TTVpostfile[:TTVpostfile.find('_')] ### filename is something like KOI-41.03_TTV_posteriors.csv, so this should isolate the KOI name.
		print(KOI)
		try:
			TTVpost = pandas.read_csv(TTVpostdir+'/'+TTVpostfile)
		except:
			print('there was a problem opening this file.')
			time.sleep(5)
			continue

		TTVpostcols = TTVpost.columns
		column_prefixes = []
		for TTVpostcol in TTVpostcols:
			TTVpostcol_prefix = TTVpostcol[:TTVpostcol.find('_')] ### will isolate seg0, seg1, ... seg11, etc.
			column_prefixes.append(TTVpostcol_prefix)
		column_prefixes = np.array(column_prefixes)
		unique_column_prefixes = np.unique(column_prefixes)
		nsegs = len(unique_column_prefixes)


		try:
			for seg in np.arange(0,nsegs,1):
				seg_RpRstar_post = np.array(TTVpost['seg'+str(seg)+'_p'])
				seg_rhostar_post = np.array(TTVpost['seg'+str(seg)+'_rhostar'])
				seg_impact_post = np.array(TTVpost['seg'+str(seg)+'_b'])
				seg_pplan_post = np.array(TTVpost['seg'+str(seg)+'_Pp'])
				seg_tauref_post = np.array(TTVpost['seg'+str(seg)+'_tau_ref'])
				seg_q1_post = np.array(TTVpost['seg'+str(seg)+'_q1'])
				seg_q2_post = np.array(TTVpost['seg'+str(seg)+'_q2'])
				seg_ntaus = 0
				seg_tau_dict = {}
				for tau in np.arange(0,20,1): 
					try:
						seg_tau_dict[tau] = np.array(TTVpost['seg'+str(seg)+'_tau'+str(tau)])
					except:
						break
				seg_likelihood_post = np.array(TTVpost['seg'+str(seg)+'_likelihood'])

				#### now we want to pull out ndraws-worth of indices
				seg_draw_idxs = np.random.randint(low=0, high=len(seg_RpRstar_post), size=ndraws)

				seg_median_pplan = np.nanmedian(seg_pplan_post[seg_draw_idxs])
				seg_median_tauref = np.nanmedian(seg_tauref_post[seg_draw_idxs])
				seg_median_taus = []
				seg_median_tau_stds = []
				for tau in seg_tau_dict.keys():
					seg_median_taus.append(np.nanmedian(seg_tau_dict[tau][seg_draw_idxs]))
					seg_median_tau_stds.append(np.nanstd(seg_tau_dict[tau][seg_draw_idxs]))
				seg_median_taus = np.array(seg_median_taus)
				seg_median_tau_stds = np.array(seg_median_tau_stds)

				#### now we have all the transit timings... want to fit a line to these.
				###### CAREFUL! THESE ARE NOT NECESSARILY UNIFORMLY SPACED!
				#### COMPUTE THE EPOCH NUMBER BASED ON IT'S DISTANCE FROM TAUREF AND THE SEGMENT PERIOD!
				###### for example: if tauref = 0, period = 35, and tau1 = 71, then let (tau1 - tauref) // 35 == 2. EPOCH 2.
				#seg_median_tau_epochs = (seg_median_taus - seg_median_tauref) // seg_median_pplan 
				seg_median_tau_epochs = np.around(((seg_median_taus - seg_median_tauref) / seg_median_pplan), 0)

				if show_debug_plots == 'y':
					print("seg_median_pplan = ", seg_median_pplan)
					print('seg_median_tauref = ', seg_median_tauref)
					print('seg_median_taus = ', seg_median_taus)
					print(' ')
					print('seg_median_tau_epochs ', seg_median_tau_epochs)
				#### the problem is, there aren't enough transit timings here to make a good period prediction is some cases.
				###### so what you really want to do is add all these epochs together, and fit at the very end.

				if seg == 0:
					koi_epochs = seg_median_tau_epochs
					koi_taus = seg_median_taus
					koi_tau_stds = seg_median_tau_stds
				else:
					#### append to lists above
					koi_epochs = np.concatenate((koi_epochs, seg_median_tau_epochs))
					koi_taus = np.concatenate((koi_taus, seg_median_taus))
					koi_tau_stds = np.concatenate((koi_tau_stds, seg_median_tau_stds))

			if len(koi_epochs) < 2:
				continue

			bad_tau_idxs = np.where(koi_taus < 0)[0]
			koi_epochs, koi_taus, koi_tau_stds = np.delete(koi_epochs, bad_tau_idxs), np.delete(koi_taus, bad_tau_idxs), np.delete(koi_tau_stds, bad_tau_idxs)

			koi_linfit_coefs = np.polyfit(x=koi_epochs, y=koi_taus, deg=1, w=1/koi_tau_stds)
			### sanity check the coefficients
			koi_linfit_slope, koi_linfit_intercept = koi_linfit_coefs
			print('nsegs = ', nsegs)
			print('koi_linfit_slope, expected pplan = ', koi_linfit_slope, seg_median_pplan)
			koi_linfit_function = np.poly1d(koi_linfit_coefs)
			### now compute the line value at each epoch
			koi_linear_ephemeris_timings = koi_linfit_function(koi_epochs)

			OminusC_days = koi_taus - koi_linear_ephemeris_timings
			OminusC_minutes = OminusC_days * 24 * 60 
			OminusC_minutes_errors = koi_tau_stds * 24 * 60

			if show_debug_plots == 'y':
				"""
				plt.scatter(koi_epochs, koi_taus, facecolor='LightCoral', edgecolor='k', s=20, zorder=1)
				plt.errorbar(koi_epochs, koi_taus, yerr=koi_tau_stds, ecolor='k', zorder=0, fmt='none')
				plt.plot(koi_epochs, koi_linear_ephemeris_timings, linestyle=':', color='k')
				plt.xlabel('Epoch')
				#plt.ylabel(r'$\Tau$')
				plt.show()
				"""

				### OK IT WORKS
				plt.scatter(koi_epochs, OminusC_minutes, facecolor='LightCoral', edgecolor='k', s=20, zorder=1)
				plt.errorbar(koi_epochs, OminusC_minutes, yerr=OminusC_minutes_errors, ecolor='k', zorder=0, fmt='none')
				plt.plot(koi_epochs, np.linspace(0,0,len(koi_epochs)), color='k', linestyle=':')
				plt.xlabel('Epoch')
				plt.ylabel('O - C [min]')
				plt.title(KOI)
				plt.show()


			print(' ')

			GOSE_TTV_summaryfile = open('/data/tethys/Documents/Central_Data/GOSE_TTV_summaryfile.csv', mode='a')
			print('writing to file.')
			for epoch, koi_epoch in enumerate(koi_epochs):
				#GOSE_TTV_summaryfile.write('KOI,Pplan_pub,Pplan_fit,epoch,tau_expected,tau_fit,OC_min,OCmin_err\n')
				GOSE_TTV_summaryfile.write(str(KOI)+','+str(seg_median_pplan)+','+str(koi_linfit_slope)+','+str(koi_epoch)+','+str(koi_linear_ephemeris_timings[epoch])+','+str(koi_taus[epoch])+','+str(OminusC_minutes[epoch])+','+str(OminusC_minutes_errors[epoch])+'\n')
			
			GOSE_TTV_summaryfile.close()

		except:
			print("unable to open this planet file. skipping.")
			time.sleep(5)
			continue


except:
	traceback.print_exc()

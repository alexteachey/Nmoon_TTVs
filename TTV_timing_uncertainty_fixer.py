from __future__ import division
import numpy as np
import pandas
import matplotlib.pyplot as plt
import os
import time
import traceback
from astropy.constants import R_sun, R_earth, M_sun, M_earth, G
import pickle
from mr_forecast import Mstat2R

#### THIS CODE WILL MAKE ADJUSTMENT TO THE sim_TTVs files.
##### originally, the files assumed a timing uncertainty equal to 4% of the transit duration.
###### this is not the way to do it.
######## Holczer 2016 uses timing uncertainty = 100 minute / single transit SNR.
########## we should use the same to compare apples to apples.
############# we'll approximate the 'signal' as width (duration) times depth (Rp/Rstar)^2.
############### the noise should be the photometric noise of the star. We should draw this from a distribution.
############# let's just do a solar approximation. 

"""
We're gonna grab the system properties from the RUN4_sim_model_settings.
AND we'll use the RUN4_sim_TTVs -- take the tdisps, and RE-NOISE them based on the the Holczer 2016 timing error formula (below)
We'll call the Sun a 15th magnitude star (very typical for the Kepler sample). What's the photometric errors on that?
Then calculate SNR, a la Holczer 2016, by doing

single transit SNR = (transit depth / photometric error) * sqrt(nobs in transit)

HOLCZER TIMING ERROR FORMULA IS (100 minutes) / single transit SNR 


"""

try:

	def afromP(period_seconds, M1_kg, M2_kg):
		numerator = G.value * period_seconds**2 * (M1_kg + M2_kg)
		denominator = 4 * np.pi**2
		final_value = (numerator / denominator)**(1/3)
		return final_value 


	show_debug_plots = input('Show plots for debugging? y/n: ')

	externaldir = '/run/media/amteachey/Auddy_Akiti/Teachey/Nmoon_TTVs'
	nsims = len(os.listdir(externaldir+'/RUN4_sim_TTVs'))
	new_RUN4_sim_TTVs_dir = externaldir+'/RUN4_FIXED_sim_TTVs'


	for sim in np.arange(0,nsims,1)+1:
		print('writing a new file for sim '+str(sim))
		TTVfilename = externaldir+'/RUN4_sim_TTVs/TTVsim'+str(sim)+'_TTVs.csv'
		modeldict = pickle.load(open(externaldir+'/RUN4_sim_model_settings/TTVsim'+str(sim)+'_system_dictionary.pkl', 'rb'))
		planet = modeldict['Planet']
		planet_mass_kg = planet['m']
		planet_mass_mearth = planet_mass_kg / M_earth.value
		planet_radius_rearth = Mstat2R(planet_mass_mearth, 0.001)[0]
		planet_radius_meters = planet_radius_rearth * R_earth.value
		planet_period_seconds = planet['P']
		planet_sma_meters = afromP(planet_period_seconds, M_sun.value, planet_mass_kg)
		tan_theta = R_sun.value / planet_sma_meters #### angle of half transit time
		theta = np.arctan(tan_theta) 
		Tdur_over_Pplan = theta / (np.pi) #### 2*theta is the full angle of the transit. (Tdur/Pplan) = (2*theta / 2*np.pi), so the RHS simplifies.
		Tdur_seconds = Tdur_over_Pplan * planet_period_seconds
		Tdur_minutes = Tdur_seconds / 60 
		Tdur_nobs = Tdur_minutes / 30 #### if Tdur_minutes = 60, Tdur_nobs = 2... 2 observations.

		#### compute the transit depth of this planet based on the sun
		transit_depth = (planet_radius_meters / R_sun.value)**2
		transit_depth_ppm = transit_depth * 1e6
		single_obs_sigphot = 350 #### estimated photometric uncertainty, in ppm, of a single Kepler observation (15th magnitude) 
		transit_SNR = (transit_depth_ppm / single_obs_sigphot) * np.sqrt(Tdur_nobs)

		transit_timing_uncertainty_minutes = 100 / transit_SNR 
		transit_timing_uncertainty_seconds = transit_timing_uncertainty_minutes * 60 

		print('transit duration [minutes]: ', Tdur_minutes)
		print('transit SNR: ', transit_SNR)
		print('transit timing uncertainty [seconds]: ', transit_timing_uncertainty_seconds)


		#### we need to re-do the timing, the noising, the fitting.
		TTVfile = pandas.read_csv(TTVfilename)
		epoch = np.array(TTVfile['epoch']).astype(int)
		xdisp = np.array(TTVfile['xdisp']).astype(float)
		tdisp = np.array(TTVfile['tdisp']).astype(float) #### RAW TIMING DISPLACEMENTS FROM GROUND TRUTH LINEAR EPHEMERIS
		noisy_tdisp = np.array(TTVfile['noisy_tdisp']).astype(float) #### TO BE REPLACED
		tobs = np.array(TTVfile['tobs']).astype(float) #### TO BE REPLACED
		TTVob = np.array(TTVfile['TTVob']).astype(float) ##### TO BE REPLACED
		timing_error = np.array(TTVfile['timing_error']).astype(float) #### TO BE REPLACED

		new_noisy_tdisp = np.random.normal(loc=tdisp, scale=transit_timing_uncertainty_seconds)

		##### NOW YOU NEED TO MAKE RAW TIMINGS
		tau0 = 0 
		new_tobs = tau0 + (epoch * planet_period_seconds) + new_noisy_tdisp

		##### now need to fit a line to these!
		weights_array = 1 / np.linspace(transit_timing_uncertainty_seconds, transit_timing_uncertainty_seconds, len(epoch))
		linefit_coefs = np.polyfit(x=epoch, y=new_tobs, deg=1, w=weights_array)
		linefit_polyfunc = np.poly1d(linefit_coefs)

		linefit_times = linefit_polyfunc(epoch) 

		#### subtract off this line to get the INFERRED LINEAR EPHEMERIS (new_TTVob)
		new_TTVob = new_tobs - linefit_times #### OBSERVED MINUS CALCULATED
		new_timing_error = np.linspace(transit_timing_uncertainty_seconds, transit_timing_uncertainty_seconds, len(timing_error))

		if show_debug_plots == 'y':
			fig, (ax1, ax2) = plt.subplots(2, sharex=True)
			ax1.scatter(epoch, new_tobs, facecolor='DodgerBlue', edgecolor='k', s=20, zorder=1)
			ax1.errorbar(epoch, tobs, yerr=new_timing_error, ecolor='k', fmt='none', zorder=0)
			ax1.plot(epoch, linefit_times, color='LightCoral', linewidth=2, linestyle='--')
			ax2.scatter(epoch, new_TTVob, facecolor='LightCoral', edgecolor='k', s=20, zorder=1)
			ax2.errorbar(epoch, new_TTVob, yerr=new_timing_error, ecolor='k', fmt='none', zorder=0)
			ax2.plot(epoch, np.linspace(0,0,len(epoch)), color='k', linestyle='--')
			ax2.set_xlabel('Epoch')
			plt.show()


		new_TTVfile = open(new_RUN4_sim_TTVs_dir+'/TTVsim'+str(sim)+'_TTVs.csv', mode='w')
		new_TTVfile.write('epoch,xdisp,tdisp,noisy_tdisp,tobs,TTVob,timing_error\n')

		for ep,xd,td,ntd,to,To,te in zip(epoch, xdisp, tdisp, new_noisy_tdisp, new_tobs, new_TTVob, new_timing_error):
			new_TTVfile.write(str(ep)+','+str(xd)+','+str(td)+','+str(ntd)+','+str(to)+','+str(To)+','+str(te)+'\n')
		new_TTVfile.close()

		print('writing finished.')
		print(' ')
		print(' ')
except:
	traceback.print_exc()
	raise Exception('something happened!')








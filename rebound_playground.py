from __future__ import division
import rebound
import reboundx
from astropy.constants import M_sun, M_jup, M_earth, G, au, R_earth, R_sun
import numpy as np
import matplotlib.pyplot as plt 
import traceback
from scipy.interpolate import interp1d
import mr_forecast as mr 
from astropy.timeseries import LombScargle 
import os
import pickle
import socket
from spock import FeatureClassifier
import time



"""
This code will simulate planets and a system of 1 to 5 moons.

The motivation is to see what happens to the TTVs of planets hosting two,
three, four major moons? What about resonances? What sort of patterns might we see in the TTVs?


YOU SHOULD REFER THE FOLLOWING PAGES, WHICH ARE DIRECTLY RELEVANT:
https://rebound.readthedocs.io/en/latest/ipython/Resonances_of_Jupiters_moons.html

AND

https://rebound.readthedocs.io/en/latest/ipython/TransitTimingVariations.html


"""



def Roche(rsat, mplan, msat):
	return rsat * ((2*mplan) / msat)**(1/3)

def RHill(sma, mplan, mstar):
	return sma * (mplan / (3*mstar))**(1/3)

def AMDcalc(Mplan_real, msats_real, asats_real, ecc_sats, inc_sats):
	Lambda_sats = msats_real * np.sqrt(G.value * Mplan_real * asats_real)
	sqrt_sats = np.sqrt(1 - ecc_sats**2)
	cosinc_sats = np.cos(inc_sats)
	argument = Lambda_sats * (1 - (sqrt_sats * cosinc_sats))
	AMD = np.nansum(argument)
	return AMD


#make_imaginary_system = input('Do you want to make an imaginary system? y/n: ')
make_imaginary_system = 'y'

debug_mode = input("Do you want to run in debug mode (won't save)? y/n: ")
if debug_mode == 'y':
	slow_it_down = input('Do you want to slow down the printouts? y/n: ')
	plot_schematic = input('Do you want to see the system schematic? y/n: ')
else:
	plot_schematic = 'n'
	slow_it_down = 'n'
how_many_systems = int(input('How many systems do you want to make? '))
use_MMR = input('Do you want to use simulate mean motion resonances? y/n: ')
enforce_stability = input('Enforce stability? y/n: ')
include_J2 = input('Do you want to include the quadrupole moment J2? y/n: ')

if socket.gethostname() == 'tethys.asiaa.sinica.edu.tw':
	#projectdir = '/data/tethys/Documents/Projects/NMoon_TTVs'
	projectdir = '/run/media/amteachey/Auddy_Akiti/Teachey/Nmoon_TTVs'
elif socket.gethostname() == 'Alexs-MacBook-Pro.local':
	projectdir = '/Users/hal9000/Documents/Projects/Nmoon_TTVsim'
else:
	projectdir = input('Please input the project directory: ')

positions_dir = projectdir+'/sim_positions'
TTV_dir = projectdir+'/sim_TTVs'
periodogram_dir = projectdir+'/sim_periodograms'
model_settings_dir = projectdir+'/sim_model_settings'
plotdir = projectdir+'/sim_plots'
simpickledir = projectdir+'/sim_pickles'

use_spock_scaling = input('Do you want to scale for SPOCK classifier? (Mp = 1, P_1 = 1, etc)? y/n: ')


nsystems_made = len(os.listdir(periodogram_dir))
nscrapped = 0
sim_number = nsystems_made + 1
print('number of systems made so far: ', nsystems_made)
print('starting sim number: ', sim_number)


sim_summary_filename = 'simulation_summary.csv'
if os.path.exists(projectdir+'/'+sim_summary_filename):
	### open and append
	pass
else:
	sim_summaryfile = open(projectdir+'/'+sim_summary_filename, mode='w')
	### write the columns!
	colnames = 'sim,nmoons,Pplan_days,ntransits,Mmoons_over_Mplan,TTV_rmsamp_sec,TTVperiod_epochs,MEGNO,SPOCK_survprop,final_AMD\n'
	sim_summaryfile.write(colnames)
	sim_summaryfile.close()


keep_going = 'y'
try:

	while nsystems_made < how_many_systems:

		if keep_going == 'n':
			raise Exception("A keyboard interruption occurred.")
			break

		unstable = 'n' #### initialize assuming stability, until shown otherwise.

		try:
			running_period_list = np.load(projectdir+'/running_list_of_derived_TTV_periods.npy').tolist()
		except:
			running_period_list = []		

		### create a simulations instance.
		sim = rebound.Simulation()
		if include_J2 == 'y':
			rebx = reboundx.Extras(sim)
			gh = rebx.load_force('gravitational_harmonics')
			rebx.add_force(gh)
		stability_model = FeatureClassifier()

		### put these in familiar units
		sim.G = G.value
		sim.units = ('s', 'm', 'kg') ### mks 


		#### now let's add Jupiter, Io, Europa, Ganymede, and Callisto!

		### Jupiter

		if make_imaginary_system == 'n':
			### MAKE JUPITER!
			sim.add(m=M_jup.value, hash='Jupiter')
			system_dict = {}
			moon_dict = {} ### make a dictionary
			moon_dict['Io'] = {'m':8.9298e22, 'a':4.218e8, 'e':0.0041, 'inc':0.050} ### make a sub-dictionary for each moon
			moon_dict['Europa'] = {'m':4.7987e22, 'a':6.711e8, 'e':0.0094, 'inc':0.471}
			moon_dict['Ganymede'] = {'m':1.4815e23, 'a':1.0704e9, 'e':0.001, 'inc':0.204}
			moon_dict['Callisto'] = {'m':1.0757e23, 'a':1.8827e9, 'e':0.0074, 'inc':0.205}




		else:
			### MASS SELECTIONS
			M_europa = 4.7987e22 ### kg
			planet_mass_kg_range = np.logspace(np.log10(10*M_earth.value), np.log10(10*M_jup.value), 10000) 
			#planet_mass_kg = np.random.choice(planet_mass_kg_range)
			planet_mass_kg = M_jup.value
			planet_mass_msol = planet_mass_kg / M_sun.value 
			planet_mass_mearth = planet_mass_kg / M_earth.value 

			#moon_mass_range = np.linspace(M_europa, M_earth.value, 10000)
			moon_mass_range = np.linspace(planet_mass_kg*1e-5, planet_mass_kg*1e-2, 10000)			

			planet_radius_rearth_median, planet_radius_rearth_plus, planet_radius_rearth_minus = mr.Mstat2R(mean=planet_mass_mearth, std=0.0)
			planet_radius_rearth = np.random.normal(loc=planet_radius_rearth_median, scale=planet_radius_rearth_plus)
			planet_radius_meters = planet_radius_rearth * R_earth.value

			if use_spock_scaling == 'y':
				sim.add(m=1, hash='planet') #### planet mass is 1, everything else will be in terms of this mass.
				#print("planet mass = ", 1)
			else:
				sim.add(m=planet_mass_kg, hash='planet')
				#print('planet mass = ', planet_mass_kg)


			Pplan_range_days = np.logspace(np.log10(10),np.log10(1500), 10000)
			Pplan_days = np.random.choice(Pplan_range_days)
			Pplan_seconds = Pplan_days * (24 * 60 * 60) 
			aplan_meters = ((Pplan_seconds**2 * (G.value * (M_sun.value + planet_mass_kg))) / (4 * np.pi**2))**(1/3)
			vorb_plan_ms = (2*np.pi * aplan_meters) / Pplan_seconds 	
			Tdur_seconds = (Pplan_seconds / np.pi) * np.arcsin(np.sqrt((R_sun.value + planet_radius_meters)**2) / aplan_meters)

			print('Mp (Mearth) = ', planet_mass_mearth)
			print('Rp (Rearth) = ',planet_radius_rearth)
			print('aplan (AU) = ', aplan_meters / au.value)
			print('Period (days) = ', Pplan_days)
			print('Tdur (hours) = ', Tdur_seconds / (60 * 60))
			#print(" ")



			Plan_Rhill_meters = RHill(sma=aplan_meters, mplan=planet_mass_kg, mstar=M_sun.value)
			Plan_Rhill_Rp = Plan_Rhill_meters / planet_radius_meters 

			moon_dict = {}
			system_dict = {}
			moon_labels = np.array(['I', 'II', 'III', 'IV', 'V'])
			system_labels = np.array(['Planet', 'I', 'II', 'III', 'IV', 'V'])
			#outer_moon = 'III' #### FOR SPOCK TESTING PURPOSES.
			#outer_moon_idx = 2
			#nmoons = 3
			outer_moon = np.random.choice(['I', 'II', 'III', 'IV', 'V'])
			outer_moon_idx = np.where(outer_moon == moon_labels)[0]
			nmoons = outer_moon_idx + 1 ### if outer_moon is III, index is 2, so nmoons = 3


			system_dict['Planet'] = {'m':planet_mass_kg, 'a':None, 'aRp':None, 'e':None, 'inc':None, 'pomega':None, 'f':None, 'P':Pplan_seconds, 'RHill':Plan_Rhill_meters}

			total_moon_mass_kg = 0
			

			used_ratios = []

			for nmoon in np.arange(0,nmoons,1):
				### select m, a, e, and inc!
				moon_label = moon_labels[nmoon]
				moon_mass_kg = np.random.choice(moon_mass_range) ### MODIFY THIS!!!!!

				#while (moon_mass_kg > 1e-4*planet_mass_kg) or (moon_mass_kg < 1e-7 * planet_mass_kg):
				#	#### draw again!
				#	moon_mass_kg = np.random.choice(moon_mass_range)

				### add to the running total!
				total_moon_mass_kg += moon_mass_kg

				moon_mass_mearth = moon_mass_kg / M_earth.value
				moon_mass_mparent = moon_mass_kg / planet_mass_kg 

				### calculate a moon_radius from forecaster!
				try:
					moon_radius_rearth_median, moon_radius_rearth_plus, moon_radius_rearth_minus = mr.Mstat2R(mean=moon_mass_mearth, std=0.01*moon_mass_mearth)
				
				except:
					total_moon_mass_kg -= moon_mass_kg #### revert back to the previous sum
					#### draw again!
					moon_mass_kg = np.random.choice(moon_mass_range)

					### add to the running total!
					total_moon_mass_kg += moon_mass_kg

					moon_mass_mearth = moon_mass_kg / M_earth.value
					moon_mass_mparent = moon_mass_kg / planet_mass_kg 		
					moon_radius_rearth_median, moon_radius_rearth_plus, moon_radius_rearth_minus = mr.Mstat2R(mean=moon_mass_mearth, std=0.01*moon_mass_mearth)			

				moon_radius_rearth = np.random.normal(loc=moon_radius_rearth_median, scale=moon_radius_rearth_plus)
				moon_radius_meters = moon_radius_rearth * R_earth.value 


				### orbital properties
				#moon_ecc = np.random.choice(np.linspace(0,0.01,10000)) #### modify this!
				moon_ecc = 0.0
				moon_inc = 0.0 ### modify this!!!!
				long_peri = np.random.choice(np.linspace(0,2*np.pi,10000))
				true_anom = np.random.choice(np.linspace(0,2*np.pi,10000))


				### calculate the Roche Limit! -- it's moon-dependent
				Roche_meters = Roche(rsat=moon_radius_meters, mplan=planet_mass_kg, msat=moon_mass_kg)
				Roche_Rp = Roche_meters / planet_radius_meters 


				if nmoon == 0:

					nsections = nmoons 
					full_range = (0.4895*Plan_Rhill_meters) - Roche_meters ### full size of the stability regime
					section_range = full_range / nsections  ### the size of each section, based on number of moons
					section_limits = np.arange(Roche_meters, 0.4895*Plan_Rhill_meters + section_range, section_range)
					assert len(section_limits) >= nsections + 1 
					
					### new December 2020 -- each moon gets a SECTION -- the moon will placed randomly in that section
					#### the number of sections = the number of moons
					inner_limit_meters = section_limits[0]*1.5
					outer_limit_meters = section_limits[1]

					moon_sma_meters = np.random.choice(np.linspace(inner_limit_meters, outer_limit_meters, 10000))
					moon_sma_Rp = moon_sma_meters / planet_radius_meters 

					### calculate the nominal orbital period based on this moon and the primary only
					moon_period_seconds = np.sqrt( (4*np.pi**2 * moon_sma_meters**3) / (G.value * (planet_mass_kg + moon_mass_kg)) )

					### calculate the MOON'S HILL RADIUS!
					moon_Rhill_meters = RHill(sma=moon_sma_meters, mplan=moon_mass_kg, mstar=planet_mass_kg)

					### this is the first one, you can chill... just catalog it amd move on
					system_dict[moon_label] = {'m':moon_mass_kg, 'a':moon_sma_meters, 'aRp':moon_sma_Rp, 'e':moon_ecc, 'inc':moon_inc, 'pomega':long_peri, 'f':true_anom, 'P':moon_period_seconds, 'RHill':moon_Rhill_meters}
					moon_dict[moon_label] = {'m':moon_mass_kg, 'a':moon_sma_meters, 'aRp':moon_sma_Rp, 'e':moon_ecc, 'inc':moon_inc, 'pomega':long_peri, 'f':true_anom, 'P':moon_period_seconds, 'RHill':moon_Rhill_meters}


				elif nmoon > 0:
					### FOR EACH SUBSEQUENT MOON!

					system_dict_keys = list(system_dict.keys()) 
					moon_dict_keys = list(moon_dict.keys())

					if use_MMR == 'y':
						### DUMMY VALUES! JUST FOR THE WHILE STATEMENT
						moon_sma_meters = 0
						moon_sma_Rp = 0 

						nwhiles = 0
						while (moon_sma_meters < inner_limit_meters) or (moon_sma_meters > 0.48985 * Plan_Rhill_meters):
							print('while loop (moon_sma_meters)...')
							nwhiles += 1
							if nwhiles >= 100:
								break

							#### this moon must have a period with an integer ratio of period to the last
							#### ALSO MUST BE OUTSIDE ROCHE AND INSIDE 0.4895 * RHill!

							integer_choices = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
							last_integer = np.random.choice(integer_choices)
							this_integer = np.random.choice(integer_choices)
							integer_ratio = last_integer / this_integer
							
							nwhiles_inner = 0
							while (this_integer == last_integer) or (integer_ratio in used_ratios) or ((1/integer_ratio) in used_ratios):
								nwhiles_inner += 1
								if nwhiles_inner >= 100:
									nwhiles = 100
									break
								### draw again!
								this_integer = np.random.choice(integer_choices)
								integer_ratio = last_integer / this_integer
								if integer_ratio < 1: 
									#### invert it! You want the period ratio to be greater than 1, so moon 2 and beyond are farther out!
									integer_ratio = 1/integer_ratio

							used_ratios.append(integer_ratio)

							last_period = moon_dict[moon_dict_keys[-1]]['P']
							this_period = last_period * integer_ratio
							moon_period_seconds = this_period 

							### need to recompute the semimajor axis
							moon_sma_meters = ((moon_period_seconds**2 * (G.value * (planet_mass_kg + moon_mass_kg))) / (4*np.pi**2))**(1/3)
							moon_sma_Rp = moon_sma_meters / planet_radius_meters 


						if nwhiles >= 100:
							print('ran into an infinte loop. skipping.')
							continue


						moon_Rhill_meters = RHill(sma=moon_sma_meters, mplan=moon_mass_kg, mstar=planet_mass_kg)
						system_dict[moon_label] = {'m':moon_mass_kg, 'a':moon_sma_meters, 'aRp':moon_sma_Rp, 'e':moon_ecc, 'pomega':long_peri, 'f':true_anom, 'inc':moon_inc, 'P':moon_period_seconds, 'RHill':moon_Rhill_meters}
						moon_dict[moon_label] = {'m':moon_mass_kg, 'a':moon_sma_meters, 'aRp':moon_sma_Rp, 'e':moon_ecc, 'pomega':long_peri, 'f':true_anom, 'inc':moon_inc, 'P':moon_period_seconds, 'RHill':moon_Rhill_meters}

						#### update the inner_limit_meters
						#inner_limit_meters = moon_sma_meters



					elif use_MMR == 'n':

						#### FOR NON-MMR SYSTEMS
						#print('last inner limit (Rp) = ', inner_limit_meters / planet_radius_meters)
						if (0.5*Plan_Rhill_meters < inner_limit_meters) and (enforce_stability == 'y'):
							print('LAST MOON OUTSIDE HALF RHILL. SCRAPPING.')
							print(' ')
							unstable = 'y'
							print(' ')
							break
						else:
							unstable = 'n'

						moon_sma_meters = np.random.choice(np.linspace(inner_limit_meters, outer_limit_meters, 10000))
						moon_sma_Rp = moon_sma_meters / planet_radius_meters
						#print('this moon sma (Rp) = ', moon_sma_Rp)
						#### update the inner_limit_meters to be the semimajor axis of this moon.

						if debug_mode == 'y' and slow_it_down == 'y':
							time.sleep(5)

						moon_period_seconds = np.sqrt( (4*np.pi**2 * moon_sma_meters**3) / (G.value * (planet_mass_kg + moon_mass_kg)) )
						moon_Rhill_meters = RHill(sma=moon_sma_meters, mplan=moon_mass_kg, mstar=planet_mass_kg)
						this_moon_Hill_range = (moon_sma_meters - moon_Rhill_meters, moon_sma_meters + moon_Rhill_meters)
						
						system_dict[moon_label] = {'m':moon_mass_kg, 'a':moon_sma_meters, 'aRp':moon_sma_Rp, 'e':moon_ecc, 'pomega':long_peri, 'f':true_anom, 'inc':moon_inc, 'P':moon_period_seconds, 'RHill':moon_Rhill_meters}
						moon_dict[moon_label] = {'m':moon_mass_kg, 'a':moon_sma_meters, 'aRp':moon_sma_Rp, 'e':moon_ecc, 'pomega':long_peri, 'f':true_anom, 'inc':moon_inc, 'P':moon_period_seconds, 'RHill':moon_Rhill_meters}




				#### new inner_limit_meters is the moon_sma_meters -- RECOMPUTE
				inner_limit_meters = moon_sma_meters #### the new inner limit based on the moon just generated above.
				full_range = (0.4895*Plan_Rhill_meters) - inner_limit_meters  #### the full range of space leftover.
				section_range = full_range / (nmoons - nmoon) #### the size of each section.
				section_limits = np.arange(inner_limit_meters, 0.4895*Plan_Rhill_meters + section_range, section_range)

				inner_limit_meters = section_limits[0]*1.5
				outer_limit_meters = section_limits[1]


				print('Moon: ', moon_label)
				print('Msat / Mplan = ', moon_mass_kg / planet_mass_kg)
				print('Roche limit (Rp) = ', Roche_Rp)
				print('a / Rp = ', moon_sma_Rp)
				print('Hill radius (Rp) = ', Plan_Rhill_Rp)
				print(' ')
			if debug_mode == 'y' and slow_it_down == 'y':
				time.sleep(10)


		if (unstable == 'y') and (enforce_stability == 'y'):
			print('# created so far = ', nsystems_made)
			nscrapped += 1
			print('# scrapped runs = ', nscrapped)
			if debug_mode == 'y' and slow_it_down == 'y':
				time.sleep(5)
			continue


		#### relevant values are a, P, e, inc, Omega (long_asc), omega (arg_peri), pomega (long_peri), f (true_anom), M (mean_anom), l (mean_long), theta (true_long)


		#### NEED TO SPECIFY COMPLETE ORBITAL PARAMETERS! INCLUDING STARTING ANGLES!!!!
		#### SPECIFICALLY, you need Omega (long_ascending), omega(arg_peri), pomega(long_peri), f(true_anom), M(mean_anom), l(mean_long), theta(true_long)
		### Io

		for moon in moon_dict.keys():
			if use_spock_scaling == 'y':
				#### scale the satellite masses to the planet mass, and the semimajor axis to that of the innermost satellite.
				sim.add(m=moon_dict[moon]['m']/planet_mass_kg, a=moon_dict[moon]['a']/moon_dict['I']['a'], e=moon_dict[moon]['e'], pomega=moon_dict[moon]['pomega'], f=moon_dict[moon]['f'], inc=moon_dict[moon]['inc'], hash=moon)
				#print('sim mass = ', moon_dict[moon]['m']/planet_mass_kg)
				#print('sim sma = ', moon_dict[moon]['a']/moon_dict['I']['a'])
				#print(' ')

			elif use_spock_scaling == 'n':
				sim.add(m=moon_dict[moon]['m'], a=moon_dict[moon]['a'], e=moon_dict[moon]['e'], pomega=moon_dict[moon]['pomega'], f=moon_dict[moon]['f'], inc=moon_dict[moon]['inc'], hash=moon)

		sim.move_to_com() ### critical step!
		sim.init_megno() #### for testing two-moon stability.
		exit_distance = Plan_Rhill_meters / moon_dict['I']['a'] #### the Hill sphere, in units of the first moon's semimajor axis.
		sim.exit_max_distance = exit_distance 

		try:
			stability_probability = stability_model.predict_stable(sim)
			print('SPOCK stability probability: ', stability_probability)

		except KeyboardInterrupt:
			keep_going = 'n'
			break

		except Exception:
			print('CANNOT RUN SPOCK ON SYSTEMS WITH FEWER THAN THREE MOONS.')
			keep_going = 'y'
			stability_probability = np.nan 

		if (stability_probability < 0.5) and (enforce_stability == 'y'):
			print('SPOCK PREDICTS UNSTABLE SYSTEM (P_stable = '+str(stability_probability 	)+'. CONTINUE.')
			print('# created so far = ', nsystems_made)
			nscrapped += 1
			print('# scrapped runs = ', nscrapped)
			if debug_mode == 'y' and slow_it_down == 'y':
				time.sleep(5)
			continue 



		particles = sim.particles
		if include_J2 == 'y':
			particles['planet'].params['J2'] = np.random.choice(np.linspace(0.001, 0.016, 10000))
		sim.integrator = 'whfast'

		sim.dt = 60 ### one minute / 60 seconds (native time unit for the sim is seconds)

		#### now let's simulate it for one Jovian year (11.86 years) = 4332.8201
		#run_period_days = 4332.8201 ### days
		run_period_years = 10
		run_period_days = run_period_years * 365.25 ### just as a test run 
		run_period_hours = run_period_days * 24
		run_period_minutes = run_period_hours * 60
		run_period_seconds = run_period_minutes * 60

		#sim.integrate(run_period_seconds, exact_finish_time=0) 

		#particle_list = ['Jupiter', 'Io', 'Europa', 'Ganymede', 'Callisto']
		particle_list = np.concatenate((['Planet'], list(moon_dict.keys())))

		Noutputs = 10000 ### number of evaluations -- not necessarily the number of time steps from sim.dt!

		times = np.linspace(0,run_period_seconds, Noutputs)
		x = np.zeros((len(particle_list), Noutputs)) ### first index is the particle, second is the timestep *to be plotted*
		y = np.zeros((len(particle_list), Noutputs)) ### ditto above.
		z = np.zeros((len(particle_list), Noutputs)) ##### NEW DECEMBER 2020 -- recording these to track inclination changes.
		#### record eccentricities and inclinations, too!
		eccs = np.zeros((len(particle_list)-1, Noutputs)) #### because we don't record for the planet!
		incs = np.zeros((len(particle_list)-1, Noutputs))
		smas = np.zeros((len(particle_list)-1, Noutputs))
		AMDs = np.zeros(shape=Noutputs)

		unstable = 'n' #### initialize under the assumption of stability
		


		for nt,t in enumerate(times):

			#### YOU NEED TO KEEP THIS AS A FOR LOOP, EVEN THOUGH IT'S INEFFICIENT, BECAUSE
			###### THIS IS WHERE YOU CALCULATE YOUR TRANSIT TIMES!

			try:
				sim.integrate(t, exact_finish_time=1) ### integrates from previous state to new specified time.
				#sim.integrator_synchronize() 

			except KeyboardInterrupt:
				keep_going = 'n'
				break

			except Exception:
				keep_going = 'y'
				print("A PARTICLE ESCAPED. TERMINATING.")
				unstable = 'y'
				break
			megno = sim.calculate_megno()
			if (nmoons == 2) and (megno > 2.5):
				print("MEGNO = ", megno)
				print("MEGNO UNSTABLE.")
				if debug_mode == 'y' and slow_it_down == 'y':
					time.sleep(5)
				unstable = 'y'
				#### break will come below

			#print('integrating to t = ', time)
			#print('t = ', t, ' of ', len(times))


			#### TEMPORARILY DEACTIVATING THIS AS A TEST.
			
			for npart,part in enumerate(particle_list): #### for every particle in the simulation 
				x[npart][nt] = particles[npart].x
				y[npart][nt] = particles[npart].y 
				z[npart][nt] = particles[npart].z #### NEW DECEMBER 2020 
				
				if npart != 0: ### particle zero is the planet!
					eccs[npart-1][nt] = particles[npart].e #### minus one because npart = 1 for moon I, but it's index = 0 for these arrays (saving space).
					incs[npart-1][nt] = particles[npart].inc
					smas[npart-1][nt] = particles[npart].a 

					particle_CoM_r = np.sqrt(particles[npart].x**2 + particles[npart].y**2 + particles[npart].z**2)
					if particle_CoM_r < Roche_meters / system_dict['I']['a']: #### Roche limit normalized by inner semimajor axis.
						print('moon shredded. Unstable!')
						unstable = 'y'
						break


			if unstable == 'y':
				break


			### CALCULATE THE ANGULAR MOMENTUM DEFICIT
			#def AMDcalc(Mplan_real, msats_real, asats_real, ecc_sats, inc_sats):
			Mplan_real = planet_mass_kg
			msats_real = []
			for moonkey in moon_dict.keys():
				msat_real = moon_dict[moonkey]['m']
				msats_real.append(msat_real)
			asats_real = smas.T[nt] * moon_dict['I']['a'] #### needs to be the real value, not the normalized value.
			ecc_sats = eccs.T[nt]
			inc_sats = incs.T[nt] 
			AMD_nt = AMDcalc(Mplan_real=Mplan_real, msats_real=msats_real, asats_real=asats_real, ecc_sats=ecc_sats, inc_sats=inc_sats+(np.pi/2))
			AMDs[nt] = AMD_nt		






		"""
		if enforce_stability == 'y':
			for npart,part in enumerate(particle_list):
				rvals = np.sqrt((x[npart] - x[0])**2 + (y[npart] - y[0])**2)
				if (np.nanmin(rvals) < 0.5*np.nanmax(rvals)) and (part != 'Planet'):
					print('min(rvals) for ', part, ' are ', (np.nanmin(rvals)/np.nanmax(rvals))*100, '% of max(rvals).')
					#### suggests significant evolution
					print('A MOON MIGRATED >= x2, or FELL INTO THE PLANET.')
					#print(' ')
					unstable = 'y'
					if debug_mode == 'y' and slow_it_down == 'y':
						time.sleep(5)
					plt.close()
					break
		"""

		print("MEGNO = ", megno)

		if unstable == 'y':
			print('# created so far = ', nsystems_made)
			nscrapped += 1
			print('# scrapped runs = ', nscrapped)
			continue


		#plt.figure(figsize=(4,4))
		### now let's plot it, each with a different color
		if plot_schematic == 'y':
			for npart,part in enumerate(particle_list):
				### plot the x and y coordinates:
				plt.plot(x[npart], y[npart], label=particle_list[npart], alpha=0.5)

			#plt.xlim(-2e9, 2e9)
			#plt.ylim(-2e9, 2e9)
			#plt.xlim(-1.2 * np.nanmax(np.abs(x.T[0])), 1.2 * np.nanmax(np.abs(x.T[0]))) #### based on the starting positions.
			#plt.ylim(-1.2 * np.nanmax(np.abs(y.T[0])), 1.2 * np.nanmax(np.abs(y.T[0])))
			#plt.xlim(-1.2 * np.abs(x[-1][0]), 1.2 * np.abs(x[-1][0])) #### starting position of the outermost particle
			#plt.ylim(-1.2 * np.abs(x[-1][0]), 1.2*np.abs(x[-1][0])) #### use x value again, so the window is symetric.
			#plt.legend()
			#plt.show()
			#plt.tight_layout()
			#plt.savefig(plotdir+'/TTVsim'+str(sim_number)+'_system_schematic.pdf', dpi=300)
			plt.show()
			plt.close()



			fig, (ax1, ax2) = plt.subplots(2, figsize=(6,10))	
			for npart,part in enumerate(particle_list):
				if npart != 0:
					#### plot the eccentricity and inclination changes 
					ax1.plot(times, incs[npart-1], label=system_labels[npart])
					ax2.plot(times, eccs[npart-1], label=system_labels[npart])
			ax1.set_ylabel('inclination')
			ax2.set_ylabel('eccentricity')
			ax2.set_xlabel('Time')
			ax1.legend()
			ax2.legend()
			plt.show()


		#### OK, let's pull out times based an orbital period of 100 days.
		### first you want to interpolate the curve, so that you can get a displacement at any point in time.
		#### OK, but the interp1d function expects a FUNCTION, which you don't have... 
		#### so instead, you're going to need to find the nearest neighbor timesteps, and interpolate locally.

		#barycenter_transit_times = np.linspace(0,np.nanmax(times), 10) ### 100 transits during the run -- the barycenter transits like clockwork
		barycenter_transit_times = np.arange(0,np.nanmax(times),Pplan_seconds)	

		transit_x_displacements = []
		transit_y_displacements = []
		transit_z_displacements = []

		for nbtt, btt in enumerate(barycenter_transit_times):
			### find the nearest indices to this time
			nearest_match_idx = np.nanargmin(np.abs(btt - times)) ### find the closest index to the time in question.
			nearest_time_neighbor_idxs = np.arange(nearest_match_idx - 2, nearest_match_idx + 3, 1) ### grab the nearby indices.
			
			try:
				nearest_times = times[nearest_time_neighbor_idxs]

			except KeyboardInterrupt:
				keep_going = 'n'
				break

			except Exception:
				keep_going = 'y'
				nearest_time_neighbor_idxs = np.arange(nearest_time_neighbor_idxs[0],len(times),1)

			nearest_times = times[nearest_time_neighbor_idxs]
			nearest_time_xvals = x[0][nearest_time_neighbor_idxs] ### nearest x values
			nearest_time_yvals = y[0][nearest_time_neighbor_idxs] ### nearest y values
			nearest_time_zvals = z[0][nearest_time_neighbor_idxs]

			#### interpolate them!!!
			nearest_time_xinterper = interp1d(nearest_times, nearest_time_xvals) #### will take xval as arg and return yval
			nearest_time_yinterper = interp1d(nearest_times, nearest_time_yvals)
			nearest_time_zinterper = interp1d(nearest_times, nearest_time_zvals)

			interpolated_xval = nearest_time_xinterper(btt) ### precise x at time t
			interpolated_yval = nearest_time_yinterper(btt) ### precise y at time t
			interpolated_zval = nearest_time_zinterper(btt)

			if use_spock_scaling == 'y':
				#### in this case, semimajor axis of moon I is 1, so you need to convert back to meters
				interpolated_xval_physical_units = interpolated_xval * moon_dict['I']['a']  
				interpolated_yval_physical_units = interpolated_yval * moon_dict['I']['a']
				interpolated_zval_physical_units = interpolated_zval * moon_dict['I']['a']
			else:
				#### no conversion was made, so you can leave them as is
				interpolated_xval_physical_units = interpolated_xval
				interpolated_yval_physical_units = interpolated_yval
				interpolated_zval_physical_units = interpolated_zval

			#### now our convention is that the swing in the x-direction from CoM is the TTV displacement.
			#### meanwhile, the observer is considered to be in the -y direction.

			transit_x_displacements.append(interpolated_xval_physical_units)
			transit_y_displacements.append(interpolated_yval_physical_units)
			transit_z_displacements.append(interpolated_zval_physical_units)


		transit_x_displacements = np.array(transit_x_displacements)
		transit_y_displacements = np.array(transit_y_displacements)
		transit_z_displacements = np.array(transit_z_displacements)
		#print('transit_x_displacements (km) = ', transit_x_displacements/1000)
		#print('transit_y_displacements (km) = ', transit_y_displacements/1000)

		####

		#### now you can calculate a *time* separation by noting that x = vt, or t = x/v.
		### let's calculate vorb from P / 2*pi*a

		Pplan_seconds = barycenter_transit_times[1] - barycenter_transit_times[0] ### it's clockwork! This is in SECONDS.

		aplan_meters = ((Pplan_seconds**2 * (G.value * (M_sun.value + M_jup.value))) / (4 * np.pi**2))**(1/3)


		vobs_plan_ms = (2 * R_sun.value) / Tdur_seconds 
		#### NOTE THAT THE ABOVE IS NOT THE ORBITAL VELOCITY -- IT IS THE SKY PROJECTED VELOCITY, 
		#### CROSSING THE DIAMETER OF THE SUN IN A TIME EQUAL TO THE TRANSIT DURATION.

		#### NOW, the ratio between the delta_t and P should be the same as arclength l / 2*pi*a
		#### SO, delta_t should be 



		#### DON'T DO IT THIS WAY
		#vorb_plan_ms = (2*np.pi * aplan_meters) / Pplan_seconds 

		### now if x is negative, t should be negative (transits early)
		### if x is positive, t should be positive (transits late)
		### that should square with O - C (observed minus calculated) -- if observed is before calculated, it's negative!


		ground_truth_time_displacements_seconds = transit_x_displacements / vobs_plan_ms
		#### simulate measurement errors
		noisy_ground_truth_time_displacements_seconds = np.random.normal(loc=ground_truth_time_displacements_seconds, scale=0.04*Tdur_seconds)
		epochs = np.arange(0,len(ground_truth_time_displacements_seconds),1)


		#### FIRST, CONVERT THEM BACK TO RAW TRANSIT TIMES!
		### TAU_N = TAU_0 + P*N
		### let TAU_0 = 0, SO 
		### TAU_N (LINEAR EPHEMERIS) = P*N
		### ADD THAT TO EACH TIME!

		raw_timings = (Pplan_seconds * epochs) + noisy_ground_truth_time_displacements_seconds

		### Pplan_seconds is the GROUND TRUTH! raw_timings is what you would measure with your telescope.


		##### the above displacement times are GROUND TRUTH TTVs -- they are NOT the same as OBSERVED TTVs
		##### with OBSERVED, WE DON'T KNOW THE GROUND TRUTH PERIOD, SO WE HAVE TO FIT A LINE.

		linfit_weights = 1/np.linspace(0.04*Tdur_seconds, 0.04*Tdur_seconds, len(epochs))
		epoch_interp = np.linspace(epochs[0], epochs[-1], 100)

		#### fit a line to the raw timings
		raw_timings_linfit = np.polyfit(x=epochs, y=raw_timings, w=linfit_weights, deg=1)
		raw_timings_linfit_func = np.poly1d(raw_timings_linfit)
		raw_timings_linear_fit = raw_timings_linfit_func(epoch_interp) #### line fit to the raw timings!

		### fit a line to the ground truth TTVs, for visualization only.
		gt_linfit = np.polyfit(x=epochs, y=noisy_ground_truth_time_displacements_seconds, w=linfit_weights, deg=1)
		gt_linfit_func = np.poly1d(gt_linfit)
		gt_linear_fit = gt_linfit_func(epoch_interp) ### line fit to the ground truth TTVs, for VISUALIZATION ONLY!


		#### the linear fit slope here IS THE INFERRED PERIOD
		Pplan_seconds_inferred = raw_timings_linfit[0] ### the slope of the linear fit!

		print('ground truth period (days) = ', Pplan_seconds / (60 * 60 * 24))
		print('inferred period (days) = ', Pplan_seconds_inferred/ (60 * 60 * 24))
		print('truth - inferred (days)= ', (Pplan_seconds - Pplan_seconds_inferred) / (60 * 60 * 24))

		#### now subtract that line -- right?
		adjusted_displacements_seconds = raw_timings - raw_timings_linfit_func(epochs)


		#### now plot them!!!

		if debug_mode == 'n':
			### WANT TO SAVE THEM!
			fig, (ax1, ax2) = plt.subplots(2, sharex=True)
			ax1.scatter(epochs, noisy_ground_truth_time_displacements_seconds, color='LightCoral', zorder=1, edgecolor='k', s=20)
			ax1.errorbar(epochs, noisy_ground_truth_time_displacements_seconds, yerr=(0.04*Tdur_seconds), zorder=0, fmt='none', ecolor='k')
			ax1.plot(epoch_interp, gt_linear_fit, c='r', linestyle='--', linewidth=2)

			ax2.scatter(epochs, adjusted_displacements_seconds, color='DodgerBlue', edgecolor='k', zorder=1, s=20)
			ax2.plot(epochs, np.linspace(0,0,len(epochs)), color='k', linestyle=':', alpha=0.75, zorder=0)
			ax2.errorbar(epochs, adjusted_displacements_seconds, yerr=(0.04*Tdur_seconds), fmt='none', zorder=0, ecolor='k')

			ax2.set_xlabel('Epoch')
			ax1.set_ylabel('ground truth O - C [s]')
			ax2.set_ylabel('observed O - C [s]')

			#plt.show()
			#plt.tight_layout()
			plt.savefig(plotdir+'/TTVsim'+str(sim_number)+'_TTVs.pdf', dpi=200)
			plt.close()




		#### generate a lomb-scargle periodogram!!!!
		sigma = 0.04*Tdur_seconds
		period_min = 2
		period_max = 500
		#LS_frequencies, LS_powers = LombScargle(epochs, adjusted_displacements_seconds, sigma).autopower(minimum_frequency=1/period_max, maximum_frequency=1/period_min, samples_per_peak=10)
		LS_frequencies = np.logspace(np.log10(1/period_max), np.log10(1/period_min), 5000)
		LS_periods = 1 / LS_frequencies
		LS_powers = LombScargle(epochs, adjusted_displacements_seconds, sigma).power(LS_frequencies)

		best_LS_freq = LS_frequencies[np.argmax(LS_powers)]
		best_LS_period = 1/best_LS_freq

		LS = LombScargle(epochs, adjusted_displacements_seconds, sigma)
		best_fit = LS.model(epochs, best_LS_freq)

		if debug_mode == 'n':
			### WANT TO SAVE THEM!
			plt.plot(LS_periods, LS_powers, color='DodgerBlue', linewidth=2)
			plt.xscale('log')
			plt.title('best period = '+str(round(best_LS_period, 2))+' epochs')
			plt.xlabel('TTV period [epochs]')
			#plt.tight_layout()
			plt.savefig(plotdir+'/TTVsim'+str(sim_number)+'_LSperiodogram.pdf', dpi=200)
			plt.close()
			#plt.show()




		if debug_mode == 'n':
			"""
			### NOW SAVE ALL YOUR SHIT!
			projectdir = '/Users/hal9000/Documents/Projects/Nmoon_TTVsim'
			model_settings_dir = projectdir+'/sim_model_settings'
			positions_dir = projectdir+'/sim_positions'
			TTV_dir = projectdir+'/sim_TTVs'
			periodogram_dir = projectdir+'/sim_periodograms'
			plotdir = projectdir+'/plots'
			"""

			#### save the system dictionary
			f = open(model_settings_dir+'/TTVsim'+str(sim_number)+'_system_dictionary.pkl', 'wb')
			pickle.dump(system_dict,f)
			f.close()

			### save the positions arrays! (separately)
			np.save(positions_dir+'/TTVsim'+str(sim_number)+'_xpos.npy', x)
			np.save(positions_dir+'/TTVsim'+str(sim_number)+'_ypos.npy', y)
			np.save(positions_dir+'/TTVsim'+str(sim_number)+'_zpos.npy', z)
			np.save(positions_dir+'/TTVsim'+str(sim_number)+'_eccs.npy', eccs)
			np.save(positions_dir+'/TTVsim'+str(sim_number)+'_incs.npy', incs)
			np.save(positions_dir+'/TTVsim'+str(sim_number)+'_smas.npy', smas)
			np.save(positions_dir+'/TTVsim'+str(sim_number)+'_AMDs.npy', AMDs)

			### save the TTVs! generate text files for this
			TTVsimfile = open(TTV_dir+'/TTVsim'+str(sim_number)+'_TTVs.csv', mode='w')
			TTVsimfile.write('epoch,xdisp,tdisp,noisy_tdisp,tobs,TTVob,timing_error\n')
			for i in np.arange(0,len(raw_timings),1):
				TTVsimfile.write(str(i)+','+str(transit_x_displacements[i])+','+str(ground_truth_time_displacements_seconds[i])+','+str(noisy_ground_truth_time_displacements_seconds[i])+','+str(raw_timings[i])+','+str(adjusted_displacements_seconds[i])+','+str(sigma)+'\n')
			TTVsimfile.close()

			### save the periodogram! as a 2xN array
			periodogram_stack = np.vstack((LS_periods, LS_powers))
			np.save(periodogram_dir+'/TTVsim'+str(sim_number)+'_periodogram.npy', periodogram_stack)

			### save the best periodogram period to the running_period_list
			running_period_list.append(best_LS_period)

			### save it to the file
			np.save(projectdir+'/running_list_of_derived_TTV_periods.npy', np.array(running_period_list))


			#### write to sim_summaryfile!
			#sim_summaryfile = open(projectdir+'/'+sim_summary_filename, mode='w')
			### write the columns!
			#colnames = 'sim,nmoons,Pplan_days,ntransits,Mmoons_over_Mplan,TTV_rmsamp,TTVperiod\n'
			#sim_summaryfile.write(colnames)
			TTVobs_RMS = np.sqrt(np.nanmean(adjusted_displacements_seconds**2))
			Mmoons_over_Mplan = total_moon_mass_kg / planet_mass_kg
			colvals = str(sim_number)+','+str(int(nmoons))+','+str(Pplan_days)+','+str(len(raw_timings))+','+str(Mmoons_over_Mplan)+','+str(TTVobs_RMS)+','+str(best_LS_period)+','+str(megno)+','+str(stability_probability)+','+str(AMDs[-1])+'\n'

			sim_summaryfile = open(projectdir+'/'+sim_summary_filename, mode='a')
			sim_summaryfile.write(colvals)
			sim_summaryfile.close()



		nsystems_made += 1
		sim_number += 1
		print('SYSTEM COMPLETED SUCCESSFULLY.')

except KeyboardInterrupt:
	keep_going = 'n'
	print('You opted not to continue.')

except Exception:
	traceback.print_exc()
	continue_query = input('Do you want to continue? ')
	if continue_query != 'y':
		raise Exception('You opted not to continue.')











from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas
import rebound
import reboundx
from astropy.constants import M_jup, R_jup, M_sun, R_sun, au 



#### THIS SCRIPT IS GONNA DO ONE THING -- TEST THE AFFECT OF J2 CHOICES ON THE SIMULATIONS.
#### WE'RE SIMPLY GONNA STEP THROUGH DIFFERENT VALUES OF J2, SAME INITIAL CONDITIONS, FOR 1 - 5 MOONS.

#### WE WANT TO TRACK THE EVOLUTION OF THESE SYSTEMS AS A FUNCTION OF J2.

def Roche(rsat, mplan, msat):
	return rsat * ((2*mplan) / msat)**(1/3)

def RHill(sma, mplan, mstar):
	return sma * (mplan / (3*mstar))**(1/3)


plot_schematic = input('Want to plot these orbits? y/n: ')

nmoon_options = np.array([1,2,3,4,5])
J2_options = np.linspace(0.001, 0.016, 20)
mass_OoM_options = np.array([-4,-3,-2])


#for nmoon in nmoon_options:
for sma in np.arange(5,30,5):
	for J2 in J2_options:
		#for mass_OoM in mass_OoM_options:
		mas_OoM = -3 

		sim = rebound.Simulation()
		if include_J2 == 'y':
			rebx = reboundx.Extras(sim)
			gh = rebx.load_force('gravitational_harmonics')
			rebx.add_force(gh)
		stability_model = FeatureClassifier()


		aplan_meters = au.value ### 1 AU 
		planet_radius_meters = R_jup.value 
		planet_mass_kg = M_jup.value 
		amoon_meters = sma * planet_radius_meters 
		moon_mass_kg = 10**(mass_OoM) * planet_mass_kg  


		Plan_Rhill_meters = RHill(sma=aplan_meters, mplan=planet_mass_kg, mstar=M_sun.value)
		Plan_Rhill_Rp = Plan_Rhill_meters / planet_radius_meters 


		#### 					
		sim.add(m=planet_mass_kg, hash='planet') #### planet mass is 1, everything else will be in terms of this mass.
		sim.add(m=moon_mass_kg, a=amoon_meters, e=0, pomega=0, f=0, inc=0, hash='I')
		sim.move_to_com() ### critical step!
		sim.init_megno() #### for testing two-moon stability.

		exit_distance = Plan_Rhill_meters #### the Hill sphere, in units of the first moon's semimajor axis.
		sim.exit_max_distance = exit_distance 

		stability_probability = stability_model.predict_stable(sim)			

		particles = sim.particles
		particles['planet'].params['J2'] = J2
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
		#particle_list = np.concatenate((['Planet'], list(moon_dict.keys())))
		particle_list = np.array(['Planet', 'I'])

		Noutputs = 10000 ### number of evaluations -- not necessarily the number of time steps from sim.dt!

		times = np.linspace(0,run_period_seconds, Noutputs)
		x = np.zeros((len(particle_list), Noutputs)) ### first index is the particle, second is the timestep *to be plotted*
		y = np.zeros((len(particle_list), Noutputs)) ### ditto above.
		z = np.zeros((len(particle_list), Noutputs)) ##### NEW DECEMBER 2020 -- recording these to track inclination changes.
		#### record eccentricities and inclinations, too!
		eccs = np.zeros((len(particle_list)-1, Noutputs)) #### because we don't record for the planet!
		incs = np.zeros((len(particle_list)-1, Noutputs))
		smas = np.zeros((len(particle_list)-1, Noutputs))
		#AMDs = np.zeros(shape=Noutputs)

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
					if particle_CoM_r < Roche_meters: #### Roche limit normalized by inner semimajor axis.
						print('moon shredded. Unstable!')
						unstable = 'y'
						break


		#plt.figure(figsize=(4,4))
		### now let's plot it, each with a different color
		if plot_schematic == 'y':
			for npart,part in enumerate(particle_list):
				### plot the x and y coordinates:
				plt.plot(x[npart], y[npart], label=particle_list[npart], alpha=0.5, label='J2 = '+str(float(J2,4)))

	plt.legend()
	plt.show()
	plt.close()


	"""
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
	"""					


from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from astropy.constants import M_sun, M_jup, M_earth, au, G
import traceback

"""
This code operates under the *assumption* that reflex motion from multiple moons is really just the 
superposition of sinusoids* (*only sinusoids if they are on circular orbits)

YOU SHOULD REALLY VERIFY THIS ASSUMPTION

For now, let's simulate it!

"""

specify_planet_specs = input("Do you want to specify planet specs? y/n: ")
if specify_planet_specs == 'n':
	plan_dict = {'Pplan_days':365.25, 'aplan_meters':au.value, 'mplan_kg':M_jup.value, 'mstar_kg':M_sun.value}
elif specify_planet_specs == 'y':
	Pplan_days = float(input("What's the orbital period, in days? "))
	Pplan_secs = Pplan_days * 24 * 60 * 60
	aplan_meters = ((Pplan_secs**2 * (G.value * (mstar+kg + mplan_kg))) / (4 * np.pi**2))**(1/3)

	### calculate the semimajor axis based on Kepler's Third Law

	#aplan_AU = float(input("What's the semimajor axis, in AU? "))
	plan_dict = {'Pplan_days':Pplan_days, 'aplan_meters':aplan_AU*au, 'mplan_kg':M_jup.value, 'mstar_kg':M_sun.value}


def get_cmap(n, name='viridis'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def calcCoM(mass_array, position_array, ndim=1):
	"""
	This function takes arrays of masses and positions and returns the center of mass
	if ndim = 1, it assumes everything's in a straight line (1-dimensional)
	elif ndim = 2 or 3, position_array should have shape (2, len(mass_array)) or (3, len(mass_array))

	arguments: mass_array, position_array

	returns: CoM float (ndim = 1) or CoM array (ndim = 2 or 3)
	
	"""

	assert len(mass_array) == len(position_array)

	if ndim == 1:
		numerator = np.nansum(mass_array * position_array) ### should be element-wise multiplication
		denominator = np.nansum(mass_array)
		CoM = numerator / denominator ### single value


	else:
		CoM_list = []
		for dim in np.arange(0,ndim,1):
			numerator = np.nansum(mass_array * position_array[dim]) ### should be element-wise multiplication
			denominator = np.nansum(mass_array)
			CoM = numerator / denominator ### single value
			CoM_list.append(CoM)
		CoM = np.array(CoM_list)

	return CoM 


def pos_P(mass_array, pos_array, CoM=None, ndim=1, subtract_CoM=True):
	"""
	Determines the position of the planet, based on the positions of the satellites.

	Planet mass must be the first element of the mass_array. pos_array should be len(mass_array) - 1 (pos_P is unknown!)
	CoM is an arbitrary value, but should be on the scale of the positional units.
	FOR EXAMPLE, if your positions are in km, so should CoM be. If NONE, CoM will be calculated from pos_array

	position of the planet is given by
	xp = {[(mp + m1 + m2)CoMx] - m1x1 - m2x2} / mp

	"""
	assert len(pos_array) == len(mass_array) - 1
	assert mass_array[0] == np.nanmax(mass_array) ### make sure the first item in the list is the planet! (largest mass)

	if CoM == None:
		CoM_val = 10*np.nanmax(np.abs(pos_array)) ### need absolute value so that CoM_value is always positive! 
	else:
		CoM_val = CoM

	first_term = np.nansum(mass_array) * CoM_val 
	second_term = np.nansum(mass_array[1:] * pos_array)
	numerator = first_term - second_term 
	denominator = mass_array[0] 
	plan_pos = numerator / denominator 

	if subtract_CoM == True:
		### subtract the CoM, so that the CoM is now ZERO! The planet position will be negative or positive, based on its position
		plan_pos = plan_pos - CoM_val 

	return plan_pos 



def moon_pos(sma, angfreq, evaltime, phase, dimension='x'):
	if dimension == 'x':
		moon_outpos = sma * np.cos(angfreq*evaltime + phase)
	elif dimension == 'y':
		moon_outpos = sma * np.sin(angfreq*evaltime + phase)

	return moon_outpos




def planTTV(ntransits, period_days=None, sma=None, date_offset=0):
	if period_days == None:
		period_days = planet_dict['Pplan_days']

	if sma == None:
		sma = planet_dict['aplan_meters']

	Pplan_secs = period_days * 24 * 60 * 60
	vorb_circ = (2*np.pi*sma) / Pplan_secs 

	print('vorb_circ [m/s] = ', vorb_circ)

	plan_toffsets = []

	for transit in np.arange(0,ntransits,1):
		### calculate the time from zero, based on the period of the planet
		transit_day = period_days * transit + date_offset ### time of barycenter midtransit, in days
		transit_second = transit_day * 24 * 60 * 60 ### time of barycenter midtransit, in seconds

		mass_array_input = [M_jup.value]

		xposs_array_input = [] ### array of moon x-positions
		yposs_array_input = [] ### array of moon y-positions
		zposs_array_input = [] ### array of moon z-positions

		for moon in moon_dict.keys():
			mass_array_input.append(moon_dict[moon]['m'])

			### calculate the angular frequency, in inverse seconds
			moon_Psecs = moon_dict[moon]['Pdays'] * 24 * 60 * 60
			moon_dict[moon]['Psecs'] = moon_Psecs
			moon_angfreq = (2*np.pi) / moon_Psecs
			moon_dict[moon]['angfreq'] = moon_angfreq

			moon_xpos = moon_pos(moon_dict[moon]['a'], moon_angfreq, transit_second, moon_dict[moon]['start_angle'], dimension='x')
			moon_ypos = moon_pos(moon_dict[moon]['a'], moon_angfreq, transit_second, moon_dict[moon]['start_angle'], dimension='y')
			moon_zpos = 0 ### for starters.		

			xposs_array_input.append(moon_xpos)
			yposs_array_input.append(moon_ypos)
			zposs_array_input.append(moon_zpos)

		mass_array_input = np.array(mass_array_input)
		xposs_array_input, yposs_array_input, zposs_array_input = np.array(xposs_array_input), np.array(yposs_array_input), np.array(zposs_array_input)

		##### now calculate the planet position
		plan_x = pos_P(mass_array_input, xposs_array_input)
		plan_y = pos_P(mass_array_input, yposs_array_input)
		plan_z = pos_P(mass_array_input, zposs_array_input)

		print('plan_x offset (meters) = ', plan_x)
		plan_toffset = plan_x / vorb_circ ### seconds
		print('plan_toffset (seconds) = ', plan_toffset)
		plan_toffsets.append(plan_toffset)

	plan_toffsets = np.array(plan_toffsets)	

	return plan_toffsets 




"""

What we will want to do is initialize a system (a planet with two or more moons.)
Each of these objects will have a positional array.
These positional arrays are updated at each time step.

How's is this not just a n-body integrator then?

The moon's motions will just be simulated as sinusoids going around the barycenter (that's the point they actually orbit, obviously)

Think of the the planet as just a counterweight -- so the moons offsets are analytical (ignoring self-interactions).

We know them precisely as a function of time.

So initialize the moons with their positions simply given by sines and cosines, and then just compute WHERE THE PLANET HAS TO BE 
TO BALANCE THE FORCE EQUATION???


"""

### now let's try it!


"""
NOTE THAT THIS randomized start angle for each moon is inconsistent with the presence of MMR.
YOU WILL WANT TO FIX THIS, BUT FOR NOW IT'S JUST A PLAYGROUND.
The start angle array is the position of the moon in its orbit -- phase=0 means starts at x=a, y=0.
"""
start_angle_array = np.linspace(-2*np.pi, 2*np.pi, 10000)
planet_mass = plan_dict['mplan_kg']
moon_dict = {} ### intialize a dictionary of moon values
moon_dict['Io'] = {'m':8.9298e22, 'Pdays':1.769, 'a':4.218e8, 'e':0.0041, 'inc':0.050, 'start_angle':np.random.choice(start_angle_array)}
#moon_dict['Europa'] = {'m':4.7987e22, 'Pdays':3.551, 'a':6.711e8, 'e':0.0094, 'inc':0.471, 'start_angle':np.random.choice(start_angle_array)}
#moon_dict['Ganymede'] = {'m':1.4815e23, 'Pdays':7.155, 'a':1.0704e9, 'e':0.0011, 'inc':0.204, 'start_angle':np.random.choice(start_angle_array)}
#moon_dict['Callisto'] = {'m':1.0757e23, 'Pdays':16.69, 'a':1.8827e9, 'e':0.0074, 'inc':0.205, 'start_angle':np.random.choice(start_angle_array)}


"""
now if we initialize these moons with arbitrary initial conditions (not actually physical), 
We can just give them some initial angle, calculate x, y, and z positions for them based on sinusoids, 
and them calculate the offset of the planet.

Let's let x be in the direction of the planet's transit, y be radial to the star, and z be up and down motion (=0 for coplanar orbits)
"""


integration_years = 0.05
integration_days = integration_years * 365.25
integration_hours = integration_days * 24
integration_minutes = integration_hours * 60
integration_seconds = integration_minutes * 60

integration_times_seconds = np.linspace(0,integration_seconds,100)
integration_times_days = integration_times_seconds / (60 * 60 * 24) 


try:
	for moon in moon_dict.keys():
		### calculate the angular frequency, in inverse seconds
		moon_Psecs = moon_dict[moon]['Pdays'] * 24 * 60 * 60
		moon_dict[moon]['Psecs'] = moon_Psecs
		moon_angfreq = (2*np.pi) / moon_Psecs
		moon_dict[moon]['angfreq'] = moon_angfreq

		print('integrating moon: ', moon)
		### calculate x,y,z positions based on sinsuoids
		moon_xyz = []
		for nit, it in enumerate(integration_times_seconds):
			print('timestep = ', nit, 'of', len(integration_times_seconds))
			moon_xpos = moon_pos(moon_dict[moon]['a'], moon_angfreq, it, moon_dict[moon]['start_angle'], dimension='x')
			moon_ypos = moon_pos(moon_dict[moon]['a'], moon_angfreq, it, moon_dict[moon]['start_angle'], dimension='y')
			moon_zpos = 0 ### for starters.
			moon_xyz.append((moon_xpos, moon_ypos, moon_zpos))

		moon_xyz = np.array(moon_xyz) ### should be an array of shape (len(integration_times), ndim)
		print(' ')
		print(' ')
		moon_dict[moon]['moon_xyz'] = moon_xyz
except:
	traceback.print_exc()



try:
	#### ok, now we should have xyz positions for each moon.
	#### let's plot them and see how they look

	fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
	cmap = get_cmap(len(moon_dict.keys()))
	for nmoon, moon in enumerate(moon_dict.keys()):
		moon_xvals = moon_dict[moon]['moon_xyz'].T[0]
		moon_yvals = moon_dict[moon]['moon_xyz'].T[1]
		moon_zvals = moon_dict[moon]['moon_xyz'].T[2]

		ax1.plot(integration_times_days, moon_xvals, color=cmap(nmoon))
		ax2.plot(integration_times_days, moon_yvals, color=cmap(nmoon))

	ax1.set_ylabel('x-displacement')
	ax2.set_ylabel('y-displacement')
	ax2.set_xlabel('Days')
	plt.show()

except:
	traceback.print_exc()


### seems to work -- now let's calculate JUPITER'S OFFSET!
#Jup_xpos = pos_P(mass_array, pos_array, CoM=None, ndim=1, subtract_CoM=True):

try:
	plan_xpos = []
	plan_ypos = []
	plan_zpos = []

	mass_array_input = [M_jup.value]
	for moon in moon_dict.keys():
		mass_array_input.append(moon_dict[moon]['m'])


	for nit, it in enumerate(integration_times_days):
		xposs_array_input = []
		yposs_array_input = []
		zposs_array_input = []

		for moon in moon_dict.keys():
			#### add the position of the moon at this point.
			xposs_array_input.append(moon_dict[moon]['moon_xyz'][nit][0])
			yposs_array_input.append(moon_dict[moon]['moon_xyz'][nit][1])
			zposs_array_input.append(moon_dict[moon]['moon_xyz'][nit][2])

		xposs_array_input = np.array(xposs_array_input)
		yposs_array_input = np.array(yposs_array_input)
		zposs_array_input = np.array(zposs_array_input)

		### now calculate the position of the planet
		plan_x = pos_P(mass_array_input, xposs_array_input)
		plan_y = pos_P(mass_array_input, yposs_array_input)
		plan_z = pos_P(mass_array_input, zposs_array_input)

		plan_xpos.append(plan_x)
		plan_ypos.append(plan_y)
		plan_zpos.append(plan_z)

	plan_xpos = np.array(plan_xpos)
	plan_ypos = np.array(plan_ypos)
	plan_zpos = np.array(plan_zpos)


	### plot it
	fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
	ax1.plot(integration_times_days, plan_xpos, color='DodgerBlue')
	ax1.set_ylabel('X-displacement')
	ax2.plot(integration_times_days, plan_ypos, color='DodgerBlue')
	ax2.set_ylabel('Y-displacement')
	ax2.set_xlabel('Days')
	plt.show()


except:
	traceback.print_exc()




#### NOW LET'S PLOT THE MOONS AND THE PLANET ALL TOGETHER!
for nmoon, moon in enumerate(moon_dict.keys()):
	moon_xvals = moon_dict[moon]['moon_xyz'].T[0]
	moon_yvals = moon_dict[moon]['moon_xyz'].T[1]
	moon_zvals = moon_dict[moon]['moon_xyz'].T[2]

	print('len(moon_xvals) = ', len(moon_xvals))
	plt.scatter(moon_xvals, moon_yvals, color=cmap(nmoon), s=np.linspace(0,50,len(moon_xvals)), alpha=0.5)
	plt.scatter(moon_xvals[0], moon_yvals[0], marker='X', color=cmap(nmoon), s=200)
	plt.scatter(moon_xvals[-1], moon_yvals[-1], marker='X', color=cmap(nmoon), s=200)
	plt.scatter(0, 0, c='k', marker='x', s=30)

	### plot a dotted line of the same color between the moon and the planet
	start_moon_line_x = np.linspace(moon_xvals[0], plan_xpos[0], 100)
	start_moon_line_y = np.linspace(moon_yvals[0], plan_ypos[0], 100)
	plt.plot(start_moon_line_x, start_moon_line_y, linestyle=':', color=cmap(nmoon), linewidth=2)

	stop_moon_line_x = np.linspace(moon_xvals[-1], plan_xpos[-1], 100)
	stop_moon_line_y = np.linspace(moon_yvals[-1], plan_ypos[-1], 100)
	plt.plot(stop_moon_line_x, stop_moon_line_y, linestyle=':', color=cmap(nmoon), linewidth=2)


plt.scatter(plan_xpos, plan_ypos, c='r', alpha=0.5, s=np.linspace(0,50,len(plan_xpos)))
plt.scatter(plan_xpos[0], plan_ypos[0], color='r', alpha=0.5, marker='X', s=200)
plt.scatter(plan_xpos[-1], plan_ypos[-1], color='r', alpha=0.5, marker='X', s=200)
plt.show()










#### now let's take these x-displacements and calculate a time offset, based on vorb
#### first we need to calculate vorb!



### hold on, what we really want to do is just evaluate the moon_positions, and planet positions, arbitrarily forward in time, without
### having to do an integration. You can just plug in the evaluation time.


ntransits = int(input('How many transits do you want to simulate? '))
epochs = np.arange(0,ntransits,1)


### now let's plot these TTVs!
try:
	OminusC_vals = planTTV(ntransits, plan_dict['Pplan_days'], plan_dict['aplan_meters'])
except:
	traceback.print_exc()



plt.scatter(epochs, OminusC_vals, s=30)
plt.xlabel('Epoch')
plt.ylabel('O - C [seconds]')
plt.show()





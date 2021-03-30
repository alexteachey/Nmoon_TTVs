from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import os
import traceback
import matplotlib.cm as cm 

try:

	#### this script will animate the systems you've generated with REBOUND

	sim_prefix = input('What is the sim prefix? ')
	projectdir = '/run/media/amteachey/Auddy_Akiti/Teachey/Nmoon_TTVs'
	if len(sim_prefix) == 0:
		positionsdir = projectdir+'/sim_positions'
	else:
		positionsdir = projectdir+'/'+sim_prefix+'_sim_positions'

	plot_individual = input('Do you want to plot an individual sim? y/n: ')
	if plot_individual == 'y':
		simnum = int(input("What simulation do you want? "))
		sims = np.array([simnum])
	else:
		sims = np.arange(1,50001,1)

	##### make the animation

	for sim in sims:
		
		#### load positions file
		sim_xpos, sim_ypos = np.load(positionsdir+'/TTVsim'+str(sim)+'_xpos.npy'), np.load(positionsdir+'/TTVsim'+str(sim)+'_ypos.npy')
		nparticles = sim_xpos.shape[0]
		max_x = np.nanmax(np.abs(sim_xpos))
		max_y = np.nanmax(np.abs(sim_ypos))
		

		fig = plt.figure(figsize=(8,8))
		ax = plt.axes(xlim=(-1.1*max_x, 1.1*max_x), ylim=(-1.1*max_y, 1.1*max_y))
		#line, = ax.plot([], [], lw=2)
		#line = ax.scatter([], [])

		"""
		def init():
			line.set_data([], [])
			#line = ([], [])
			#line = ax.scatter([], [])
			#return line,
			return line,
		"""

		#### compute the angular sweep of each particle, per time step
		delta_xs = sim_xpos.T[1] - sim_xpos.T[0]
		delta_ys = sim_ypos.T[1] - sim_ypos.T[0]
		angular_sweeps = np.arctan(delta_ys / delta_xs)  #### inner moons will have larger angular sweeps... so you want your tolerance to be within the smaller sweep

		def animate(i):
			x = sim_xpos.T[i]
			y = sim_ypos.T[i]

			colors = cm.get_cmap('viridis')(np.linspace(0,1,6))

			#### find angle crossings

			angles = np.arctan(y/x) #### equal to the number of particles
			sizes = np.linspace(10,10,len(x))
			these_colors = colors[:len(x)]


			if len(x) > 2:
				#### if the angles of any two pairs is less than say, 2 degrees, change the color to red and make the size=20
				for nang1, ang1 in enumerate(angles):
					if nang1 == 0:
						continue
					#for nang2, ang2 in enumerate(angles[nang1+1:]):
					nang2 = nang1+1 					
					try:
						ang2 = angles[nang2]
						#### they also need to be in the same quadrant!
						#if (np.sign(x[nang1]) == np.sign(x[nang2])) and (np.sign(y[nang1]) == np.sign(y[nang2])) and (np.abs(ang1 - ang2) <= angular_sweeps[nang2]):
						if (np.sign(x[nang1]) == np.sign(x[nang2])) and (np.sign(y[nang1]) == np.sign(y[nang2])) and (np.abs(ang1 - ang2) <= 2 * (np.pi/180)):

							sizes[nang1] = 30
							sizes[nang2] = 30
							these_colors[nang1] = np.array([1.,0.,0.,1.]) ### inner hit, make it red
							these_colors[nang2] = np.array([0.,0.,1.,1.]) ### outer hit, make it blue
							ax.plot([x[nang1], x[nang2]], [y[nang1], y[nang2]], c='k', linestyle=':', alpha=0.5)
					except:
						break


			#colors = cm.viridis(0,1,6)

			#line.set_data(x,y)
			line = ax.scatter(x, y, c=these_colors, s=sizes, alpha=0.7)
			#return line,
			#return line

		#anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(sim_xpos[0]), interval=40, blit=True)
		anim = animation.FuncAnimation(fig, animate, frames=len(sim_xpos[0]), interval=20, cache_frame_data=False, blit=False)			

		plt.show()
		


		"""
		fig, ax = plt.subplots(figsize=(5, 3))
		ax.set(xlim=(-max_x, max_x), ylim=(-max_y, max_y))


		#### FROM THE ONLINE EXAMPLE

		x = np.linspace(-3, 3, 91)
		t = np.linspace(1, 25, 30)
		X2, T2 = np.meshgrid(x, t)
		sinT2 = np.sin(2*np.pi*T2/T2.max())
		F = 0.9*sinT2*np.sinc(X2*(1 + sinT2))
		line = ax.plot(x, F[0, :], color='k', lw=2)[0]
		def animate(i):
    		line.set_ydata(F[i, :])
    	anim = FuncAnimation(fig, animate, interval=100, frames=len(t)-1)
 
		plt.draw()
		plt.show()
		"""


		"""

		scat = ax.scatter(sim_xpos.T[0], sim_ypos.T[0])
 
		def animate(i):
			#y_i = F[i, ::3]
			#scat.set_offsets(np.c_[x[::3], y_i])
			#scat.set_offsets(np.c_[sim_xpos[::3], sim_ypos[::3]])
			scat.set_offsets(np.c_[sim_xpos.T[i], sim_ypos.T[i]])

		plt.draw()
		plt.show()
		"""





except:
	traceback.print_exc()
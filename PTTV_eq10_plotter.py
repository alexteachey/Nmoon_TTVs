from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

plt.rcParams["font.family"] = 'serif'


#### this code will make a heatmap of equation 10 of the Exomoon Corridor result https://arxiv.org/pdf/2012.00764.pdf

Pplan_range = np.logspace(np.log10(10), np.log10(1500), 1000) #### days
Psat_range = np.linspace(2,50,500) #### days

#### can't do it like this, because there will be invalid numbers. DO IT AS A FRACTION OF HILL SPHERE.
fRHill_range = np.arange(0.05,0.55, 0.05)


def PTTV_eq10(Pplan, Psat):
	numerator = 1 
	denom_first_term = 1 / Psat
	denom_second_term = np.round(Pplan / Psat) * (1/Pplan)
	denom = denom_first_term - denom_second_term
	output = numerator / denom 
	if np.isfinite(output) == False:
		output = np.nan

	if output < 0:
		print('OUTPUT NEGATIVE! ')
		print('Pp, Ps = ', Pplan, Psat)
		print(' ')
	return output 


Psat_vs_Pplan_days = np.zeros(shape=(500,1000))
Psat_vs_Pplan_epochs = np.zeros(shape=(500,1000))
#fRHill_vs_Pplan_days = np.zeros(shape=(len(fRHill_range),100))
#fRHill_vs_Pplan_epochs = np.zeros(shape=(len(fRHill_range),100))

for nPp,Pp in enumerate(Pplan_range):
	for nPs,Ps in enumerate(Psat_range):
	#for nfRHill,fRHill in enumerate(fRHill_range):
		#### CALCULATE Ps!!!!
		#Ps = Pp * np.sqrt((fRHill**3) / 3)

		PTTV_days = PTTV_eq10(Pp, Ps)
		PTTV_epochs = PTTV_days / Pp 

		#fRHill_vs_Pplan_days[nfRHill][nPp] = PTTV_days
		#fRHill_vs_Pplan_epochs[nfRHill][nPp] = PTTV_epochs
		Psat_vs_Pplan_days[nPs][nPp] = PTTV_days
		Psat_vs_Pplan_epochs[nPs][nPp] = PTTV_epochs

##### set all nonzero_values to np.nan!

#Psat_vs_Pplan_days[Psat_vs_Pplan_days < 0] = np.nan
#Psat_vs_Pplan_epochs[Psat_vs_Pplan_epochs < 0] = np.nan
Psat_vs_Pplan_days = np.abs(Psat_vs_Pplan_days)
Psat_vs_Pplan_epochs = np.abs(Psat_vs_Pplan_epochs)

plt.imshow(Psat_vs_Pplan_epochs, aspect='auto', vmin=0, vmax=30, origin='lower', cmap=cm.viridis, interpolation='antialiased')
plt.xticks(ticks=np.linspace(0,len(Pplan_range),10), labels=np.round(np.logspace(np.log10(np.nanmin(Pplan_range)), np.log10(np.nanmax(Pplan_range)),10)))
plt.xlabel(r'$P_P$ [days]')
plt.yticks(ticks=np.linspace(0,len(Psat_range),10), labels=np.round(np.linspace(np.nanmin(Psat_range), np.nanmax(Psat_range),10)))
plt.ylabel(r'$P_S$ [days]')
plt.colorbar(label=r'$P_{\mathrm{TTV}}$')
#plt.imshow(fRHill_vs_Pplan_epochs, origin='lower')
plt.show()

plt.imshow(Psat_vs_Pplan_epochs, aspect='auto', vmin=0, vmax=30, origin='lower', cmap=cm.viridis, interpolation='nearest')
plt.xticks(ticks=np.linspace(0,len(Pplan_range),10), labels=np.round(np.logspace(np.log10(np.nanmin(Pplan_range)), np.log10(np.nanmax(Pplan_range)),10)))
plt.xlabel(r'$P_P$ [days]')
plt.yticks(ticks=np.linspace(0,len(Psat_range),10), labels=np.round(np.linspace(np.nanmin(Psat_range), np.nanmax(Psat_range),10)))
plt.ylabel(r'$P_S$ [days]')
plt.colorbar(label='Epochs')
#plt.imshow(fRHill_vs_Pplan_epochs, origin='lower')
plt.show()


flattened_Psat_vs_Pplan_epochs = np.ndarray.flatten(Psat_vs_Pplan_epochs)
n, bins, edges = plt.hist(flattened_Psat_vs_Pplan_epochs, bins=np.linspace(2,30,15), facecolor='DodgerBlue', edgecolor='k', alpha=0.7)
plt.xlabel(r'$P_{\mathrm{TTV}}$')
plt.show()
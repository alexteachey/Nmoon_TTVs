from __future__ import division
import numpy as np
import matplotlib.pyplot as plt 
import pandas
import matplotlib.cm as cm

### THIS SCRIPT WILL INTERPRET THE Multi-Layer Perceptron Results 

projectdir = '/Users/hal9000/Documents/Projects/Nmoon_TTVsim'

MLPfile = pandas.read_csv(projectdir+'/keras_MLP_run.csv')

MLPcols = MLPfile.columns

#### generate a scatter plot of precision and recall as a function of # of layers and # of 


nmoon_options = np.array([2,3,4,5])

for nmoons in nmoon_options:
	nlayers = np.array(MLPfile['num_layers']).astype(float)
	neurperlayer = np.array(MLPfile['neurons_per_layer']).astype(float)
	npreds = np.array(MLPfile['n'+str(nmoons)+'_preds']).astype(float)
	nactual = np.array(MLPfile['n'+str(nmoons)+'_actual']).astype(float)

	precision = np.array(MLPfile['n'+str(nmoons)+'_precision']).astype(float)
	recall = np.array(MLPfile['n'+str(nmoons)+'_recall']).astype(float)


	#### PLOTTING PRECISION AND RECALL AS A FUNCTION OF THE ANN ARCHITECTURE.
	fig, (ax1, ax2) = plt.subplots(2, sharex=True)

	im1 = ax1.scatter(nlayers, neurperlayer, c=precision, s=20, cmap='coolwarm', vmin=0, vmax=1)
	ax1.set_ylabel('neurons / layer')
	im2 = ax2.scatter(nlayers, neurperlayer, c=recall, s=20, cmap='coolwarm', vmin=0, vmax=1)
	ax2.set_ylabel('neurons / layer')
	ax2.set_xlabel('# hidden layers')
	#fig.colorbar(im1)
	#fig.colorbar(im2)
	ax1.set_title('Precision, '+str(nmoons)+' moons')
	ax2.set_title('Recall, '+str(nmoons)+' moons')
	plt.show()


	#### make a histogram of npreds / nactual.
	histbins = np.logspace(-1,1,20)
	n, bins, edges = plt.hist(npreds/nactual, bins=histbins, facecolor='DodgerBlue', edgecolor='k')
	plt.xlabel('# predicted / # actual')
	plt.xscale('log')
	plt.title("# moons = "+str(nmoons))
	plt.show()



	#### plot these results as a function of nlayers
	plt.scatter(nlayers, npreds/nactual, c=precision, s=20, cmap='coolwarm')
	plt.plot(np.linspace(np.nanmin(nlayers), np.nanmax(nlayers), 100), np.linspace(1,1,100), c='k', linestyle='--', alpha=0.5)
	plt.xlabel('# layers')
	plt.ylabel('# predicted / # actual')
	plt.ylim(10**(-3), 10**3)	
	plt.yscale('log')
	plt.title('# moons = '+str(nmoons))
	plt.show()	



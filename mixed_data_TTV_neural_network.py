from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas
import os 
import time
import traceback
import pickle
#from tensorflow import keras
from scipy.optimize import curve_fit
from astropy.timeseries import LombScargle 
from scipy.interpolate import interp1d 
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from astropy.constants import R_sun, G, M_sun
from keras.models import Sequential, Model

from keras.layers import Dense, BatchNormalization, Conv1D, MaxPooling1D, AveragePooling1D, Activation, Dropout, Flatten, Input, concatenate
from keras.utils import to_categorical
import socket
from keras.callbacks import EarlyStopping, ModelCheckpoint
import sys



def local_variables_return():
	exception_type, exception_value, traceback_msg = sys.exc_info()
	if traceback_msg is not None:
		previous = traceback_msg
		current = traceback_msg.tb_next
		while current is not None:
			previous = current
			current = current.tb_next
		return previous.tb_frame.f_locals






callbacks_list = [EarlyStopping(monitor='val_accuracy', patience=5)]



#### THIS SCRIPT WILL READ IN SIMULATED OBSERVATIONS FOR AN EFFORT AT MIXED DATA CLASSIFICATION / REGRESSION.
###### THE PROBLEM IS SIMPLE -- based on observations, can we regress the number of moons present?
######## THE PLAN IS TO UTILIZE *BOTH* numerical values in an Artificial Neural Network, *AND* a CNN for the periodogram.
########## WITH THIS MIXED DATA NEURAL NETWORK, HOPEFULLY THERE IS ENOUGH INFORMATION TO INFER THE NUMBER OF MOONS.
############# BUT, THIS IS A HARD PROBLEM. ALL OF THE MOON PERIODICITIES ARE BELOW THE NYQUIST RATE, SO WE ARE ONLY SEEING ALIASES.
############### THE QUESTION IS, CAN WE INFER THE PRESENCE OF N > 1 FROM THOSE ALIASES?

try:
	if socket.gethostname() == 'tethys.asiaa.sinica.edu.tw':
		#projectdir = '/data/tethys/Documents/Projects/NMoon_TTVs'
		projectdir = '/run/media/amteachey/Auddy_Akiti/Teachey/Nmoon_TTVs'
	elif socket.gethostname() == 'Alexs-MacBook-Pro.local':
		projectdir = '/Users/hal9000/Documents/Projects/Nmoon_TTVsim'
	else:
		projectdir = input('Please input the project directory: ')

	positionsdir = projectdir+'/sim_positions'
	ttvfiledir = projectdir+'/sim_TTVs'
	LSdir = projectdir+'/sim_periodograms'
	modeldictdir = projectdir+'/sim_model_settings'
	plotdir = projectdir+'/sim_plots'


	load_simobsdict = input('Do you want to load the simulated observation dictionary? y/n: ')


	keras_or_skl = input("Do you want to use 'k'eras or 's'cikit-learn? ")
	if keras_or_skl == 's':
		mlp_or_rf = input("Do you want to use a 'm'ulti-layer perceptron, or a 'r'andom forest classifier? ")
	elif keras_or_skl == 'k':
		add_CNN = input("Do you want to add in a CNN for the periodogram? y/n: ")

	normalize_data = input('Do you want to normalize neural network inputs? (recommended): ')
	require_TTV_evidence = input('DO you want to require EVIDENCE FOR TTVs (deltaBIC <= -2)? y/n: ')
	run_second_validation = input('Run second validation? y/n: ')
	if run_second_validation == 'n': 
		print('NOTE: code will not be saving the scores and model hyperparameters. Monitor in real time.')
		time.sleep(5)


	if load_simobsdict == 'y':
		try:
			print('loading dictionaries....')
			simobs_dict = pickle.load(open(projectdir+'/simobs_dictionary.pkl', "rb"))
			parameter_dict = pickle.load(open(projectdir+'/sim_parameters_dictionary.pkl', 'rb'))
			print('loaded.')

		except:
			print('could not load simobs_dictionary.pkl. Reading in fresh...')
			load_simobsdict = 'n' #### unable to load it, will have to read in manually.

		try:
			periodogram_stack = np.load(projectdir+'/periodogram_stack.npy')
		except:
			print('COULD NOT LOAD periodogramstack. Will read in fresh.')

	if load_simobsdict == 'n':
		simobsfile = pandas.read_csv(projectdir+'/simulated_observations.csv')
		sims = np.array(simobsfile['sim']).astype(int)
		simobs_columns = simobsfile.columns
		simobs_dict = {}
		parameter_dict = {}

		#### first generate the parameter dict, which will be a dictionary listed by column name
		for col in simobs_columns:
			parameter_dict[col] = np.array(simobsfile[col])


		#### now generate a dictionary indexed by the simulation name -- so you can pair these numbers with the periodograms.
		for nsim,sim in enumerate(sims):
			print('reading in sim ', sim)
			simobs_dict[sim] = {}
			for col in simobs_columns:
				if col != 'sim':
					simobs_dict[sim][col] = np.array(simobsfile[col][nsim])

			#### now load the periodogram! and add it to the same dictionary
			simobs_dict[sim]['periodogram'] = np.load(LSdir+'/TTVsim'+str(sim)+'_periodogram.npy')

		#### save the dictionaries!
		pickle.dump(simobs_dict, open(projectdir+'/simobs_dictionary.pkl', 'wb'))	
		pickle.dump(parameter_dict, open(projectdir+'/sim_parameters_dictionary.pkl', 'wb'))

	### NOW YOU SHOULD HAVE simobs_dict, indexed by simulation number, and parameter_dict, indexed by the parameter name.





	#### now let's start with some function definitions that we'll need

	def normalize(array):
		num = array - np.nanmin(array)
		denom = np.nanmax(array) - np.nanmin(array)
		return num / denom 


	def build_MLP_inputs(*args, arrays='features'):
		### we're going to use this function to build a 2-D input_array
		### in a vertical stack, the shape = (nrows, ncolumns)
		#### for the MLP classifier, it has to be shape = n_samples, n_features
		##### that is, each ROW is a training example (sample), and each COLUMN is an input feature.
		###### what we're going to be LOADING IN, THOUGH, ARE ARRAYS OF FEATURES.
		###### THE LENGTH WILL BE EQUAL TO THE NUMBER OF SAMPLES (EXAMPLES)

		if arrays == 'features':
			#### means each array input is something like an array of Mplans, Pplans, etc.
			outstack = np.zeros(shape=(len(args[0]), len(args)))
			for narg, arg in enumerate(args):
				outstack.T[narg] = arg ### transposing so you can index by the column number

		elif arrays == 'examples':
			#### means each array is a series of features for a single input example. would be weird to do it this way, but I guess you could.
			outstack = np.zeros(shape=(len(args), len(args[0])))
			for narg, arg in enumerate(args):
				outstack[narg] = arg 

		return outstack


	def build_CNN_inputs(data_stack, output_classes):
		### we're going to use this function to build a 2-D input_array
		### in a vertical stack, the shape = (nrows, ncolumns)
		#### for the MLP classifier, it has to be shape = n_samples, n_features
		##### that is, each ROW is a training example (sample), and each COLUMN is an input feature.
		###### what we're going to be LOADING IN, THOUGH, ARE ARRAYS OF FEATURES.
		###### THE LENGTH WILL BE EQUAL TO THE NUMBER OF SAMPLES (EXAMPLES)

		data_stack = np.expand_dims(data_stack, axis=2)
		output_classes = np.expand_dims(output_classes, axis=1)

		return data_stack, output_classes 




	def MLP_classifier(input_array, target_classifications, hidden_layers=5, neurons_per_layer=100, validation_fraction=0.1):
		##### SCIKIT-LEARN FRAMEWORK
		#### input_array should be 2-Dimensional (shape=n_samples, n_features)
		assert input_array.shape[0] > input_array.shape[1] 
		#### you're gonna want more examples than features!

		assert len(target_classifications) == input_array.shape[0] ### every input sample should have a corresponding classification output!

		hidden_layer_neuron_list = []
		for i in np.arange(0,hidden_layers,1):
			hidden_layer_neuron_list.append(neurons_per_layer)
		hidden_layer_tuple = tuple(hidden_layer_neuron_list)

		clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=hidden_layer_tuple, verbose=True, early_stopping=True, validation_fraction=validation_fraction)

		clf.fit(input_array, target_classifications)

		return clf #### outputs the classifier that's ready to take inputs as, inputs.


	def RF_classifier(input_array, target_classifications, n_estimators=100, max_depth=10, max_features=5):
		#### SCIKIT-lEARN FRAMEWORK
		assert input_array.shape[0] > input_array.shape[1]
		assert len(target_classifications) == input_array.shape[0]

		clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features)
		clf.fit(input_array, target_classifications)

		return clf 




	#### DEFINITIONS #####
	def create_CNN(input_array_length, num_mlp_outputs=None, nfilters=4, kernel_size=3, pool_size=5, pool_type='avg', strides=2, dropout=0.25, nconv_layers=5, ndense=4, input_shape=None, regress=False, num_classes=5):
		#### KERAS FRAMEWORK
		if num_mlp_outputs == None:
			num_mlp_outputs = num_classes+1

		#### FOLLOWING THIS TUTORIAL FOR MIXED DATA ANN / CNN: https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/
		inputShape = (input_array_length, 1) ####  image is 1D, and there's just one channel (no color information)
		chanDim = -1

		print('inputShape = ', inputShape)

		### define the model input
		inputs = Input(shape=inputShape)

		for convlayer in np.arange(0,nconv_layers,1):
			if convlayer == 0:
				x = inputs
			multiplier = 2**convlayer
			try:
				x = Conv1D(filters=multiplier*nfilters, kernel_size=tuple([kernel_size]), padding='same', kernel_initializer='orthogonal')(x) ### 16 outputs.
				x = Conv1D(filters=multiplier*nfilters, kernel_size=tuple([kernel_size]), padding='same', kernel_initializer='orthogonal')(x)
				x = Activation("relu")(x)
				x = BatchNormalization(axis=chanDim)(x)
				if pool_type == 'avg':
					x = AveragePooling1D(pool_size=tuple([pool_size]), strides=tuple([strides]))(x)
				elif pool_type == 'max':
					x = MaxPooling1D(pool_size=tuple([pool_size]), strides=tuple([strides]))(x)	
			except:
				local_variables = local_variables_return()
				traceback.print_exc(limit=10)

				print("EXCEPTION OCCURRED IN BUILDING THE CNN. BREAKING THE LOOP.")
				break			

		x = Flatten()(x)
		x = Dense(nfilters)(x)
		x = Activation("relu")(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = Dropout(dropout)(x)

		x = Dense(num_mlp_outputs)(x)
		x = Activation("relu")(x)

		if regress == True:
			x = Dense(1, activation="linear")(x)
		elif regress == False:
			if num_classes > 2:
				x = Dense(num_classes+1, activation='softmax')(x)
			elif num_classes == 2:
				x = Dense(num_classes, activation='sigmoid')(x)

		model = Model(inputs, x)
		return model 
		


	def create_MLP(nhidden_layers, neurons_per_layer, input_shape, regress=False, num_classes=5):
		model = Sequential()
		for hlnum in np.arange(0,nhidden_layers,1): ### add the hidden layers  
			#### for every layer you're adding
			#model.add(Dense(nplo, input_layer=5, activation='relu'))
			if hlnum == 0:
				model.add(Dense(neurons_per_layer, activation='relu', input_shape=input_shape))
			else:
				model.add(Dense(neurons_per_layer, activation='relu'))

		if regress == True:
			model.add(Dense(1, activation='linear'))
		elif regress == False:
			if num_classes > 2:
				model.add(Dense(num_classes+1, activation='softmax'))
			elif num_classes == 2:
				model.add(Dense(num_classes, activation='sigmoid'))

		return model 


	#### DEFINITIONS END ####



	##### NOW LET'S START BUILDING THE INPUTS FOR THE ANN AND CNN!
	sim = np.array(parameter_dict['sim']).astype(int)
	nmoons = np.array(parameter_dict['Nmoons']).astype(int)
	Pplan_days = np.array(parameter_dict['Pplan_days']).astype(float)
	ntransits = np.array(parameter_dict['ntransits']).astype(int)
	TTV_rmsamp_sec = np.array(parameter_dict['TTV_rmsamp_sec']).astype(float)
	TTVperiod_epochs = np.array(parameter_dict['TTVperiod_epochs']).astype(float)
	peak_power = np.array(parameter_dict['peak_power']).astype(float)
	fit_sineamp = np.array(parameter_dict['fit_sineamp']).astype(float)
	deltaBIC = np.array(parameter_dict['deltaBIC']).astype(float)
	MEGNO = np.array(parameter_dict['MEGNO']).astype(float) #### should be around 2 for stable systems!
	SPOCKprob = np.array(parameter_dict['SPOCK_prob']).astype(float) #### will have NaNs! 

	print('# of SYSTEMS FOR N MOONS (ORIGINAL): ')
	print("N=1: ", len(np.where(nmoons == 1)[0]))
	print("N=2: ", len(np.where(nmoons == 2)[0]))
	print("N=3: ", len(np.where(nmoons == 3)[0]))
	print('N=4: ', len(np.where(nmoons == 4)[0]))
	print("N=5: ", len(np.where(nmoons == 5)[0]))

	if add_CNN == 'y':
		try:
			print('periodogram_stack.shape = ', periodogram_stack.shape)
		except:
			print("COUD NOT LOAD periodogram_stack. Loading in fresh...")

			nsims = len(simobs_dict.keys())
			for ns, s in enumerate(sim):
				if ns == 0:
					##### create the periodogram_stack
					periodogram_stack = np.zeros( shape=( nsims, len(simobs_dict[s]['periodogram'][1]) ) )
				print('loading periodogram: ', s)
				simperiodogram = simobs_dict[s]['periodogram'][1]
				periodogram_stack[ns] = simperiodogram 

			np.save(projectdir+'/periodogram_stack.npy', periodogram_stack)


	##### need to cut this down to ONLY STABLE SYSTEMS!!!!!
	good_spockprob_idxs = np.where(SPOCKprob >= 0.8)[0]
	good_SPOCKprob_MEGNOs = MEGNO[good_spockprob_idxs]
	MEGNO_twosig_lowerlim, MEGNO_twosig_upperlim = np.nanpercentile(good_SPOCKprob_MEGNOs, 2.5), np.nanpercentile(good_SPOCKprob_MEGNOs, 97.5)
	good_MEGNO_idxs = np.where((MEGNO >= MEGNO_twosig_lowerlim) & (MEGNO <= MEGNO_twosig_upperlim))[0]
	evidence_for_TTVs_idxs = np.where(deltaBIC <= -2)[0]
	stable_idxs = []
	
	#### STABILITY CHECK
	for idx in np.arange(0,len(sim),1):
		if (idx in good_spockprob_idxs):
			stable_idxs.append(idx) #### if SPOCK probability is good, we go with this
		
		elif (np.isfinite(SPOCKprob[idx]) == False) and (idx in good_MEGNO_idxs): ### if SPOCK prob is unavailable but the MEGNO is good, we go with this
			stable_idxs.append(idx) 
		
		elif (SPOCKprob[idx] < 0.9) and (idx in good_MEGNO_idxs):
			continue 

	if require_TTV_evidence == 'y':
		first_cut_idxs = np.intersect1d(stable_idxs, evidence_for_TTVs_idxs)
	elif require_TTV_evidence == 'n':
		first_cut_idxs = stable_idxs 

	print('# of systems left after first cut: ', len(first_cut_idxs))

	if add_CNN == 'y':
		periodogram_stack = periodogram_stack[first_cut_idxs]


	#### BEFORE WE REDUCE THESE SAMPLE SIZES, GENERATE A HISTOGRAM OF 1) number in each group, 2) number stable 3) number good evidence 4) final
	fig, ax = plt.subplots(4, figsize=(6,10))
	histbins = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
	n1hist = ax[0].hist(nmoons, bins=histbins, facecolor='Crimson', edgecolor='k')
	ymax = 1.1*np.nanmax(n1hist[0])
	ax[0].set_ylabel('Total')
	ax[0].set_ylim(0,ymax)
	n2hist = ax[1].hist(nmoons[stable_idxs], bins=histbins, facecolor='Gold', edgecolor='k')
	ax[1].set_ylabel('Stable')
	ax[1].set_ylim(0,ymax)
	n3hist = ax[2].hist(nmoons[evidence_for_TTVs_idxs], bins=histbins, facecolor='SeaGreen', edgecolor='k')
	ax[2].set_ylabel(r'$\Delta \mathrm{BIC} \leq -2$')
	ax[2].set_ylim(0,ymax)
	n4hist = ax[3].hist(nmoons[first_cut_idxs], bins=histbins, facecolor='SlateBlue', edgecolor='k')
	ax[3].set_ylabel('Stable w/ TTVs')
	ax[3].set_ylim(0,ymax)
	ax[3].set_xlabel('# moons')
	plt.show()


	#### CUTTING DOWN THESE ARRAYS SO THAT THEY ARE ONLY THE STABLE SYSTEMS -- IT MAKES NO SENSE TO TRAIN ON UNSTABLE SYSTEMS!!!!!!
	sim, nmoons, Pplan_days, ntransits, TTV_rmsamp_sec, TTVperiod_epochs, peak_power, fit_sineamp, deltaBIC, MEGNO, SPOCKprob = sim[first_cut_idxs], nmoons[first_cut_idxs], Pplan_days[first_cut_idxs], ntransits[first_cut_idxs], TTV_rmsamp_sec[first_cut_idxs], TTVperiod_epochs[first_cut_idxs], peak_power[first_cut_idxs], fit_sineamp[first_cut_idxs], deltaBIC[first_cut_idxs], MEGNO[first_cut_idxs], SPOCKprob[first_cut_idxs]

	print('# of SYSTEMS FOR N MOONS (AFTER STABILITY AND MAYBE TTV EVIDENCE REQ): ')
	print("N=1: ", len(np.where(nmoons == 1)[0]))
	print("N=2: ", len(np.where(nmoons == 2)[0]))
	print("N=3: ", len(np.where(nmoons == 3)[0]))
	print('N=4: ', len(np.where(nmoons == 4)[0]))
	print("N=5: ", len(np.where(nmoons == 5)[0]))


	### now we're going to UPDATE the final_idxs (can use the same code above), to BALANCE THE AND VALIDATION_SET
	n1s, n2s, n3s, n4s, n5s = 0, 0, 0, 0, 0
	choose_max_per_cat = input('Based on numbers above, do you want to choose how many samples are in each category? y/n: ')
	if choose_max_per_cat == 'n':
		max_per_category = np.nanmin((len(np.where(nmoons == 1)[0]), len(np.where(nmoons  == 2)[0]), len(np.where(nmoons == 3)[0]), len(np.where(nmoons == 4)[0]), len(np.where(nmoons == 5)[0])))
	elif choose_max_per_cat == 'y':
		max_per_category = int(input('How many systems per category? (balancing is important): '))
	else:
		try:
			max_per_category = int(choose_max_per_cat)
		except:
			raise Exception('you did not answer the question correctly.')

	second_cut_idxs = []
	for fidx, moon_num in enumerate(nmoons):
		if (moon_num == 1) and (n1s < max_per_category):
			n1s += 1
			second_cut_idxs.append(fidx)
		elif (moon_num == 2) and (n2s < max_per_category):
			n2s += 1
			second_cut_idxs.append(fidx)
		elif (moon_num == 3) and (n3s < max_per_category):
			n3s += 1
			second_cut_idxs.append(fidx)
		elif (moon_num == 4) and (n4s < max_per_category):
			n4s += 1
			second_cut_idxs.append(fidx)

		elif (moon_num == 5) and (n5s < max_per_category):
			n5s += 1
			second_cut_idxs.append(fidx)
		else:
			continue

		if (n1s == max_per_category) and (n2s == max_per_category) and (n3s == max_per_category) and (n4s == max_per_category) and (n5s == max_per_category):
			break

	#### NOW WE HAVE BALANCED TRAINING AND VALIDATION FRACTIONS, NO MATTER WHAT.
	sim, nmoons, Pplan_days, ntransits, TTV_rmsamp_sec, TTVperiod_epochs, peak_power, fit_sineamp, deltaBIC, MEGNO, SPOCKprob = sim[second_cut_idxs], nmoons[second_cut_idxs], Pplan_days[second_cut_idxs], ntransits[second_cut_idxs], TTV_rmsamp_sec[second_cut_idxs], TTVperiod_epochs[second_cut_idxs], peak_power[second_cut_idxs], fit_sineamp[second_cut_idxs], deltaBIC[second_cut_idxs], MEGNO[second_cut_idxs], SPOCKprob[second_cut_idxs]
	
	if add_CNN == 'y':
		periodogram_stack = periodogram_stack[second_cut_idxs]

	print('# of examples per category (total) = ', max_per_category)
	print('total samples available for training and validation: ', max_per_category*5) 
	time.sleep(5)



	train_fraction = 0.8
	validate_fraction = 0.2

	all_idxs = np.arange(0,len(sim),1)
	np.random.seed(42) #### standardize the shuffle
	np.random.shuffle(all_idxs) ### shuffle them so they're random order
	training_idxs = all_idxs[:int(train_fraction*len(all_idxs))]
	validation_idxs = all_idxs[int(train_fraction*len(all_idxs)):]


	if normalize_data == 'n':
		#MLP_input_array = build_MLP_inputs(planet_masses, planet_periods, TTV_periods, TTV_rms, TTV_snrs)
		MLP_input_array = build_MLP_inputs(Pplan_days, ntransits, TTV_rmsamp_sec, TTVperiod_epochs, peak_power, fit_sineamp, deltaBIC)
	
		if add_CNN == 'y':
			CNN_input_array = build_CNN_inputs(periodogram_stack, nmoons)


	elif normalize_data == 'y':
		normed_Pplan_days = normalize(Pplan_days)
		normed_ntransits = normalize(ntransits)
		normed_TTV_rmsamp_sec = normalize(TTV_rmsamp_sec)
		normed_TTVperiod_epochs = normalize(TTVperiod_epochs)
		normed_peak_power = normalize(peak_power)
		normed_fit_sineamp = normalize(fit_sineamp)
		normed_deltaBIC = normalize(deltaBIC) #### careful with this one! you have positive and negative values!
		MLP_input_array = build_MLP_inputs(normed_Pplan_days, normed_ntransits, normed_TTV_rmsamp_sec, normed_TTVperiod_epochs, normed_peak_power, normed_fit_sineamp, normed_deltaBIC)		
		for periodogram_row in np.arange(0,periodogram_stack.shape[0],1):
			periodogram_stack[periodogram_row] = normalize(periodogram_stack[periodogram_row])
		CNN_input_array = build_CNN_inputs(periodogram_stack, nmoons)


	hidden_layer_options = np.arange(1,30,1)
	#neurons_per_layer_options = np.arange(10,110,10)
	neurons_per_layer_options = np.array([100])
	n_estimator_options = np.arange(10,110,10)
	max_depth_options = np.arange(1,21,1)
	max_features_options = np.arange(2,6,1)

	#### STARTING VALUES -- WILL BE UPDATED DURING THE LOOP!
	best_hlo = 0 ### hidden layer options
	best_nplo = 0 ### neurons_per_layer options
	best_neo = 0 ### best n_estimator options
	best_mdo = 0  #### best max_depth_options
	best_mfo = 0 #### best max_features_options

	best_accuracy = 0 
	best_categorical_accuracy = 0
	best_n1_accuracy = 0
	best_n2_accuracy = 0
	best_n3_accuracy = 0
	best_n4_accuracy = 0
	best_n5_accuracy = 0

	total_or_categorical = input("Do you want to optimize 't'otal or 'c'ategorical accuracy? ")

	if keras_or_skl == 'k':
		MLP_filename = 'keras_MLP_run.csv'

	elif keras_or_skl == 's':
		if mlp_or_rf == 'm':
			MLP_filename = 'sklearn_MLP_run.csv'
		elif mlp_or_rf == 'r':
			MLP_filename = 'sklearn_RF_run.csv'

	if os.path.exists(MLP_filename):
		#### open it and read the last hidden layer and neurons_per_layer -- first and second entries
		MLPfile = pandas.read_csv(projectdir+'/'+MLP_filename)
		if (keras_or_skl == 'k') or ((keras_or_skl == 's') and (mpl_or_rf == 'm')):
			last_hl = np.array(MLPfile['num_layers'])[-1]
			last_npl = np.array(MLPfile['neurons_per_layer'])[-1]
		elif (keras_or_skl == 's') and (mpl_or_rf == 'r'):
			last_neo = np.array(MLPfile['num_estimators'])[-1]
			last_md = np.array(MLPfile['max_depth'])[-1]
			last_mf = np.array(MLPfile['max_features'])[-1]


	else:
		last_hl = -1
		last_npl = -1 
		last_neo = -1
		last_md = -1
		last_mf = -1 

		MLPfile = open(projectdir+'/'+MLP_filename, mode='w')

		if (keras_or_skl == 'k') or ((keras_or_skl == 's') and (mpl_or_rf == 'm')):
			MLPfile.write('num_layers,neurons_per_layer,total_valacc,n1_actual,n1_preds,n1_precision,n1_recall,n2_actual,n2_preds,n2_precision,n2_recall,n3_actual,n3_preds,n3_precision,n3_recall,n4_actual,n4_preds,n4_precision,n4_recall,n5_actual,n5_preds,n5_precision,n5_recall\n')
		elif (keras_or_skl == 's') and (mpl_or_rf == 'r'):
			MLPfile.write('num_estimators,max_depth,max_features,total_valacc,n1_actual,n1_preds,n1_precision,n1_recall,n2_actual,n2_preds,n2_precision,n2_recall,n3_actual,n3_preds,n3_precision,n3_recall,n4_actual,n4_preds,n4_precision,n4_recall,n5_actual,n5_preds,n5_precision,n5_recall\n')


		MLPfile.close()


	if keras_or_skl == 'k': #### keras multilayer perceptron -- takes # hidden layers and #neurons per layer
		loop1 = hidden_layer_options
		loop2 = neurons_per_layer_options
		loop3 = np.array([1])

	elif (keras_or_skl == 's') and (mpl_or_rf == 'm'): #### scikit-learn multilayer perceptron -- takes # hidden layers and #neurons per layer
		loop1 = hidden_layer_options
		loop2 = neurons_per_layer_options
		loop3 = np.array([1])

	elif (keras_or_skl == 's') and (mpl_or_rf == 'r'): #### scikit-learn random forest classifier -- takes # estimators, max depth, and max features.
		loop1 = n_estimator_options 
		loop2 = max_depth_options 
		loop3 = max_features_options


	"""
	START THE MASSIVE LOOP OF HYPERPARAMETERS!!!! 
	STARTS BELOW.
	"""

	for l1 in loop1:

		if (keras_or_skl == 'k') or ((keras_or_skl == 's') and (mpl_or_rf == 'm')):
			hlo = l1 ### hidden layer option 
			if hlo < last_hl:
				continue

		elif (keras_or_skl == 's') and (mpl_or_rf == 'r'):
			neo = l1 ### n_estimator option 
			if neo < last_neo:
				continue 

		for l2 in loop2:
			if (keras_or_skl == 'k') or ((keras_or_skl == 's') and (mpl_or_rf == 'm')):
				nplo = l2 ### neurons per layer option 
				if nplo < last_npl:
					continue

			elif (keras_or_skl == 's') and (mpl_or_rf == 'r'):
				mdo = l2 ### max_depth option
				if mdo < last_md:
					continue 


			for l3 in loop3:
				if (keras_or_skl == 'k') or ((keras_or_skl == 's') and (mpl_or_rf == 'm')):
					dummy_variable = l3 ### neurons per layer option 

				elif (keras_or_skl == 's') and (mpl_or_rf == 'r'):
					mfo = l3 ### max_features option
					if mfo < last_mf:
						continue 


				#### start a new row!
				MLPfile = open(projectdir+'/'+MLP_filename, mode='a')

				if (keras_or_skl == 'k') or ((keras_or_skl == 's') and (mpl_or_rf == 'm')):
					print('Number of hidden layers = ', hlo)
					print('Number of neurons per layer = ', nplo)

				elif (keras_or_skl == 's') and (mpl_or_rf == 'r'):
					print('Number of estimators = ', neo)
					print('Maximum depth = ', mdo)
					print("Maximum features = ", mfo)


				#time.sleep(1)

				if keras_or_skl == 's':
					### calling your own function, which fits the data under the hood.
					if mpl_or_rf == 'm':
						#classifier = MLP_classifier(MLP_input_array[training_idxs], nmoons[training_idxs], hidden_layers=hlo, neurons_per_layer=nplo)
						classifier = MLP_classifier(MLP_input_array[training_idxs], nmoons[training_idxs], hidden_layers=hlo, neurons_per_layer=nplo, validation_fraction=0.1) ### under the hood validation
					
					elif mpl_or_rf == 'r':
						classifier = RF_classifier(MLP_input_array[training_idxs], nmoons[training_idxs]) ### there's no validation fraction for this one


				elif keras_or_skl == 'k':
					regress_target = 'n'

					#### FIRST MODEL THE MLP
					model_MLP = create_MLP(nhidden_layers=hlo, neurons_per_layer=nplo, regress=False, num_classes=5, input_shape=tuple([MLP_input_array.shape[1]]))


					if add_CNN == 'n':
						print("MLP compiling...")
						model_MLP.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
						print('MLP compiled.')
						#### train this sucker
						MLP_history = model_MLP.fit(x=MLP_input_array[training_idxs], y=nmoons[training_idxs], verbose=1, callbacks=callbacks_list, validation_split=0.2, epochs=100, batch_size=10)

					#### ADD IN THE CNN
					elif add_CNN == 'y':

						#### NOT READY TO COMPILE AND FIT -- we're gonna add in the CNN!
						array_size = len(simobs_dict[1]['periodogram'])

						#### WILL WANT TO ITERATE ON THIS ARCHITECTURE!!!!
						model_CNN = create_CNN(input_array_length=array_size, nfilters=4, kernel_size=3, pool_size=5, pool_type='avg', strides=2, dropout=0.25, nconv_layers=5, ndense=4, input_shape=None, regress=False, num_classes=5)


						combinedInput = concatenate([model_MLP.output, model_CNN.output])

						#### NOW WE PUT THEM TOGETHER! 
						x = Dense(6, activation='relu')(combinedInput)
						if regress_target == 'y':
							x = Dense(1, activation='linear')(x)
						elif regress_target == 'n':
							x = Dense(6, activation='softmax')(x)


						final_model = Model(inputs=[model_MLP.input, model_CNN.input], outputs=x)
						if regress_target == 'n':
							final_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
						elif regress_target == 'y':
							final_model.compile(loss='mean_absolute_percentage_error', optimizer='adam', metrics=['accuracy'])

						final_model.fit(x=[MLP_input_array[training_idxs], CNN_input_array[0][training_idxs]], y=nmoons[training_idxs], verbose=1, callbacks=callbacks_list, validation_split=0.2, epochs=200, batch_size=10)







				if run_second_validation == 'y':
					### VALIDATE IT!

					nhits = 0
					nmisses = 0
					ntotal = 0 

					### categorical accuracy -- maybe optimize this instead!!!!

					###### NOTE: PRECISION = TP / (TP + FP) --> nhits = TP, npreds == TP + FP, so precision = nhits / npreds
					####### RECALL = TP / TP + FN --> nhits == TP, ntotal = TP + FN so recall = nhits / ntotal.

					n1hits, n1preds, n1total, n1_accuracy = 0, 0, 0, 0 
					n2hits, n2preds, n2total, n2_accuracy = 0, 0, 0, 0
					n3hits, n3preds, n3total, n3_accuracy = 0, 0, 0, 0
					n4hits, n4preds, n4total, n4_accuracy = 0, 0, 0, 0
					n5hits, n5preds, n5total, n5_accuracy = 0, 0, 0, 0

					for nsample, sample in enumerate(MLP_input_array[validation_idxs]):
						print('classifying: ')
						#### make a prediction!
						actual_num_moons = nmoons[validation_idxs][nsample]
						
						if keras_or_skl == 's':
							
							if mpl_or_rf == 'm':
								try:
									classification = classifier.predict(sample)[0]
								except:
									sample = sample.reshape(1,-1)
									classification = classifier.predict(sample)[0]

							elif mpl_or_rf == 'r':
								try:
									classification = classifier.predict(sample)[0]
								except:
									sample = sample.reshape(1,-1)
									classification = classifier.predict(sample)[0]


						elif keras_or_skl == 'k':
							try:
								#classification = model.predict_classes(sample)[0] ### DEPRECATED
								classification = int(np.argmax(model.predict(sample), axis=-1))
							except:
								sample = sample.reshape(1,-1)
								#classification = model.predict_classes(sample)[0]
								classification = int(np.argmax(model.predict(sample), axis=-1))



						#print('classification: ', classification)
						if classification == 1:
							n1preds += 1
						elif classification == 2:
							n2preds += 1
						elif classification == 3:
							n3preds += 1
						elif classification == 4:
							n4preds += 1
						elif classification == 5:
							n5preds += 1

						#print('actual: ', actual_num_moons)
						if classification == actual_num_moons: ### TP
							print('HIT: '+str(classification)+' classification, actual '+str(actual_num_moons))
							nhits += 1
							if actual_num_moons == 1:
								n1hits += 1
							elif actual_num_moons == 2:
								n2hits += 1
							elif actual_num_moons == 3:
								n3hits += 1
							elif actual_num_moons == 4:
								n4hits += 1
							elif actual_num_moons == 5:
								n5hits += 1

							#print('HIT!')
						else: #### 
							print("MISS: "+str(classification)+' classification, actual '+str(actual_num_moons))
							nmisses += 1
							#print("MISS!")

						if actual_num_moons == 1:
							n1total += 1
							n1_accuracy = n1hits / n1total 
						elif actual_num_moons == 2:
							n2total += 1
							n2_accuracy = n2hits / n2total
						elif actual_num_moons == 3:
							n3total += 1
							n3_accuracy = n3hits / n3total 
						elif actual_num_moons == 4:
							n4total += 1
							n4_accuracy = n4hits / n4total
						elif actual_num_moons == 5:
							n5total += 1 
							n5_accuracy = n5hits / n5total 

						ntotal += 1

						running_accuracy = nhits / ntotal
						running_categorical_accuracy = np.nanmean((n1_accuracy, n2_accuracy, n3_accuracy, n4_accuracy, n5_accuracy))

						##### END OF EXAMPLE LOOP

					try:
						n1precision = n1hits / n1preds
						n1recall = n1hits / n1total
					except:
						n1precision, n1recall = 0, 0

					try:
						n2precision = n2hits / n2preds
						n2recall = n2hits / n2total
					except:
						n2precision, n2recall = 0, 0

					try:
						n3precision = n3hits / n3preds
						n3recall = n3hits / n3total
					except:
						n3precision, n3recall = 0, 0

					try:
						n4precision = n4hits / n4preds
						n4recall = n4hits / n4total
					except:
						n4precision, n4recall = 0, 0

					try:
						n5precision = n5hits / n5preds
						n5recall = n5hits / n5total
					except:
						n5precision, n5recall = 0, 0





					accuracy = running_accuracy
					categorical_accuracy = running_categorical_accuracy 
					print('accuracy = ', accuracy*100)
					print('categorical accuracy = ', categorical_accuracy*100)
					print('n1 precision / recall = ', n1precision, n1recall)
					print('n2 precision / recall = ', n2precision, n2recall)
					print('n3 precision / recall = ', n3precision, n3recall)
					print('n4 precision / recall = ', n4precision, n4recall)
					print('n5 precision / recall = ', n5precision, n5recall)


					#### BEST VALUES UPDATES
					if accuracy > best_accuracy: ### improved 
						best_accuracy = accuracy 
						if total_or_categorical == 't':
							if (keras_or_skl == 'k') or ((keras_or_skl == 's') and (mpl_or_rf == 'm')):
								best_hlo = hlo 
								best_nplo = nplo 
							elif (keras_or_skl == 's') and (mpl_or_rf == 'r'):
								best_neo = neo 
								best_mdo = mdo 
								best_mfo = mfo 


					if categorical_accuracy > best_categorical_accuracy: ### improved 
						best_categorical_accuracy = categorical_accuracy 
						if total_or_categorical == 'c':
							if (keras_or_skl == 'k') or ((keras_or_skl == 's') and (mpl_or_rf == 'm')):
								best_hlo = hlo 
								best_nplo = nplo 
							elif (keras_or_skl == 's') and (mpl_or_rf == 'r'):
								best_neo = neo 
								best_mdo = mdo 
								best_mfo = mfo 

					if n1_accuracy > best_n1_accuracy:
						best_n1_accuracy = n1_accuracy
					if n2_accuracy > best_n2_accuracy:
						best_n2_accuracy = n2_accuracy
					if n3_accuracy > best_n3_accuracy:
						best_n3_accuracy = n3_accuracy
					if n4_accuracy > best_n4_accuracy:
						best_n4_accuracy = n4_accuracy
					if n5_accuracy > best_n5_accuracy:
						best_n5_accuracy = n5_accuracy 

		
					#MLPfile.write('num_layers,neurons_per_layer,total_valacc,n1_actual,n1_preds,n1_precision,n2_recall,n2_actual,n2_preds,n2_precision,n2_recall,n3_actual,n3_preds,n3_precision,n3_recall,n4_actual,n4_preds,n4_precision,n4_recall,n5_actual,n5_preds,n5_precision,n5_recall\n')	
					if (keras_or_skl == 'k') or ((keras_or_skl == 's') and (mpl_or_rf == 'm')):
						MLPfile.write(str(hlo)+','+str(nplo)+','+str(accuracy)+','+str(n1total)+','+str(n1preds)+','+str(n1precision)+','+str(n1recall)+','+str(n2total)+','+str(n2preds)+','+str(n2precision)+','+str(n2recall)+','+str(n3total)+','+str(n3preds)+','+str(n3precision)+','+str(n3recall)+','+str(n4total)+','+str(n4preds)+','+str(n4precision)+','+str(n4recall)+','+str(n5total)+','+str(n5preds)+','+str(n5precision)+','+str(n5recall)+'\n')

					elif (keras_or_skl == 's') and (mpl_or_rf == 'r'):
					#MLPfile.write('num_estimators,max_depth,max_features,total_valacc,n1_actual,n1_preds,n1_precision,n1_recall,n2_actual,n2_preds,n2_precision,n2_recall,n3_actual,n3_preds,n3_precision,n3_recall,n4_actual,n4_preds,n4_precision,n4_recall,n5_actual,n5_preds,n5_precision,n5_recall\n')
						MLPfile.write(str(neo)+','+str(mdo)+','+str(mfo)+','+str(accuracy)+','+str(n1total)+','+str(n1preds)+','+str(n1precision)+','+str(n1recall)+','+str(n2total)+','+str(n2preds)+','+str(n2precision)+','+str(n2recall)+','+str(n3total)+','+str(n3preds)+','+str(n3precision)+','+str(n3recall)+','+str(n4total)+','+str(n4preds)+','+str(n4precision)+','+str(n4recall)+','+str(n5total)+','+str(n5preds)+','+str(n5precision)+','+str(n5recall)+'\n')

					MLPfile.close()



				#time.sleep(5)




	print(" ")
	print(" X X X X X ")
	print(" SUMMARY ")
	print(" X X X X X ")
	print(' ')
	print('best accuracy so far: ', best_accuracy)
	print("best categorical accuracy so far: ", best_categorical_accuracy)
	print('best # hidden layers = ', best_hlo)
	print('best # neurons per layer = ', best_nplo)






	raise Exception('this is all you want to do right now.')



















except:
	local_variables = local_variables_return()
	traceback.print_exc()



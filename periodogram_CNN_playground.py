from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import traceback
import os
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, AveragePooling1D, MaxPooling1D, Dropout, Dense, Flatten
from tensorflow.keras.models import Sequential 
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas



#### THIS CODE WILL ATTEMPT TO USE A CONVOLUTIONAL NEURAL NETWORK TO CLASSIFY PERIODOGRAMS WITH 1,2,3,4,or 5 moons.
### CNN STRUCTURAL INPUTS


norm_individual_periodograms = input('Do you want to normalize periodograms on an individual basis? y/n: ')

#### DEFINITIONS #####
def createModel(nfilters=4, kernel_size=3, pool_size=5, pool_type='avg', strides=2, dropout=0.25, nconv_layers=5, ndense=4, input_shape=None, num_classes=None):
	model = Sequential()
	for convlayer in np.arange(0,nconv_layers,1):
		multiplier = 2**convlayer
		model.add(Conv1D(filters=multiplier*nfilters, kernel_size=tuple([kernel_size]), padding='same', kernel_initializer='orthogonal', activation='relu', input_shape=input_shape)) ### 16 outputs.
		model.add(Conv1D(filters=multiplier*nfilters, kernel_size=tuple([kernel_size]), padding='same', kernel_initializer='orthogonal', activation='relu'))
		if pool_type == 'avg':
			model.add(AveragePooling1D(pool_size=tuple([pool_size]), strides=tuple([strides])))
		elif pool_type == 'max':
			model.add(MaxPooling1D(pool_size=tuple([pool_size]), strides=tuple([strides])))	

	model.add(Dropout(dropout))
	multiplier = 2**(convlayer+1)
	for denselayer in np.arange(0,ndense,1):
		model.add(Dense(multiplier*nfilters, activation='relu'))

	model.add(Flatten())
	model.add(Dense(num_classes, activation='sigmoid'))

	return model
#### DEFINITIONS END ####







projectdir = '/Users/hal9000/Documents/Projects/Nmoon_TTVsim'
periodogramdir = projectdir+'/sim_periodograms'
modelsettingsdir = projectdir+'/sim_model_settings'

nsims = len(os.listdir(periodogramdir))

### need to build the batches of periodograms -- should all have same dimensions! so just load the powers, not the periods.
power_stack = np.zeros(shape=(nsims, 1000))
nmoon_list = [] #### target classification

n1idxs, n2idxs, n3idxs, n4idxs, n5idxs = [], [], [], [], []

for nsim,sim in enumerate(np.arange(2,nsims+1,1)): ### offset in the sim numbers! lol

	print('loading sim # '+str(sim))
	sim_periodogram = np.load(periodogramdir+'/TTVsim'+str(sim)+'_periodogram.npy')
	sim_periods, sim_powers = sim_periodogram 
	power_stack[nsim] = sim_powers

	if norm_individual_periodograms == 'y':
		sim_powers = (sim_powers - np.nanmin(sim_powers)) / (np.nanmax(sim_powers) - np.nanmin(sim_powers))


	##### now you need to pull out the number of moons from the dictionary!
	sim_dict = pickle.load(open(modelsettingsdir+'/TTVsim'+str(sim)+'_system_dictionary.pkl', "rb"))
	nmoons = len(sim_dict.keys()) - 1 ### the first key is the planet!!!!
	nmoon_list.append(nmoons)

	if nmoons == 1:
		n1idxs.append(nsim)
	elif nmoons == 2:
		n2idxs.append(nsim)
	elif nmoons == 3:
		n3idxs.append(nsim)
	elif nmoons == 4:
		n4idxs.append(nsim)
	elif nmoons == 5:
		n5idxs.append(nsim)

	#### need to index number of each moons, so we can build training and validation sets!


try:
	#### FINISHED LOADING IN THE DATA -- NOW LET'S SORT IT.
	nmoon_list = np.array(nmoon_list)
	n1idxs, n2idxs, n3idxs, n4idxs, n5idxs = np.array(n1idxs), np.array(n2idxs), np.array(n3idxs), np.array(n4idxs), np.array(n5idxs)

	ntraining_examples_per_class = int(np.nanmin((len(n1idxs), len(n2idxs), len(n3idxs), len(n4idxs), len(n5idxs))) / 2 )
	#### ^ We want our training set to be balanced, so we will only use as many as the least representative
	##### DIVIDE BY TWO, SINCE WE STILL NEED SOME OF THIS LEAST REPRESENTED CLASS IN THE VALIDATION SAMPLE!

	np.random.shuffle(n1idxs)
	np.random.shuffle(n2idxs)
	np.random.shuffle(n3idxs)
	np.random.shuffle(n4idxs)
	np.random.shuffle(n5idxs)

	#### grab the first ntraining_examples_per_class for the training, the rest will be held back for validation
	n1idxs_training = n1idxs[:ntraining_examples_per_class]
	n2idxs_training = n2idxs[:ntraining_examples_per_class]
	n3idxs_training = n3idxs[:ntraining_examples_per_class]
	n4idxs_training = n4idxs[:ntraining_examples_per_class]
	n5idxs_training = n5idxs[:ntraining_examples_per_class]

	n1idxs_validation = n1idxs[ntraining_examples_per_class:]
	n2idxs_validation = n2idxs[ntraining_examples_per_class:]
	n3idxs_validation = n3idxs[ntraining_examples_per_class:]
	n4idxs_validation = n4idxs[ntraining_examples_per_class:]
	n5idxs_validation = n5idxs[ntraining_examples_per_class:]

	training_idxs = np.concatenate((n1idxs_training, n2idxs_training, n3idxs_training, n4idxs_training, n5idxs_training))
	validation_idxs = np.concatenate((n1idxs_validation, n2idxs_validation, n3idxs_validation, n4idxs_validation, n5idxs_validation))

	training_data = power_stack[training_idxs]
	training_classes = nmoon_list[training_idxs]

	validation_data = power_stack[validation_idxs]
	validation_classes = nmoon_list[validation_idxs]

	#### split validation_data into 1) under the hood (uth) and playground (pg)
	uth_validation_data = validation_data[:int(0.5*len(validation_classes))]
	uth_validation_classes = validation_classes[:int(0.5*len(validation_classes))]

	pg_validation_data = validation_data[int(0.5*len(validation_classes)):]
	pg_validation_classes = validation_classes[int(0.5*len(validation_classes)):]

	#### made need to reshape these, but let's try just loading them in!
	training_data = np.expand_dims(training_data, axis=2)
	training_classes = np.expand_dims(training_classes, axis=1)
	uth_validation_data = np.expand_dims(uth_validation_data, axis=2)
	uth_validation_classes = np.expand_dims(uth_validation_classes, axis=1)
	pg_validation_data = np.expand_dims(pg_validation_data, axis=2)
	og_validation_classes = np.expand_dims(pg_validation_classes, axis=1)


	#input_shape = (len(training_data[0]), 1) 
	input_shape = (len(training_data[0]), 1)
	num_classes = 6


	### build the model!

	step = 0
	steps_to_take = 1000

	last_test_loss = np.inf 
	last_test_accuracy = 0 


	if os.path.exists(projectdir+'/Periodogram_CNN_results.csv'):
		### read it in!
		CNN_resultsfile = pandas.read_csv(projectdir+'/Periodogram_CNN_results.csv')

		step = int(np.array(CNN_resultsfile['step'])[-1])
		last_filters_input = int(np.array(CNN_resultsfile['filters'])[-1])
		last_kernel_input = int(np.array(CNN_resultsfile['kernel'])[-1])
		last_pool_input = int(np.array(CNN_resultsfile['pool_size'])[-1])
		last_pool_type_input = np.array(CNN_resultsfile['pool_type'])[-1]
		last_stride_input = int(np.array(CNN_resultsfile['stride'])[-1])
		last_dropout_input = float(np.array(CNN_resultsfile['dropout'])[-1])
		last_nconv_layers_input = int(np.array(CNN_resultsfile['nconv_layers'])[-1])
		last_ndense_layers_input = int(np.array(CNN_resultsfile['ndense_layers'])[-1])
		last_test_loss = float(np.array(CNN_resultsfile['loss'])[-1])
		last_test_accuracy = float(np.array(CNN_resultsfile['accuracy'])[-1])

	else:
		#### write it new!
		CNN_resultsfile = open(projectdir+'/Periodogram_CNN_results.csv', mode='w')
		CNN_resultsfile.write('step,filters,kernel,pool_size,pool_type,stride,dropout,nconv_layers,ndense_layers,loss,accuracy\n')
		CNN_resultsfile.close()

	while step < steps_to_take:
		try:
			#### initiate the walker

			filters_bounds = [1,20]
			kernel_bounds = [1,20]
			pool_bounds = [1,20]
			stride_bounds = [1,20]
			dropout_bounds = [0,0.4]
			nconv_layers_bounds = [1,10]
			ndense_layers_bounds = [1,10]

			if step == 0:
				#### random start!
				filters_input = np.random.randint(1,6) ### random integer between 1 and 5 (inclusive)
				kernel_input = np.random.randint(1,6) ### ditto
				pool_input = np.random.randint(1,6) 
				pool_type_input = np.random.choice(['max', 'avg'])
				stride_input = np.random.randint(1,6)
				dropout_input = np.random.choice(np.arange(0.05,0.35,0.05))
				nconv_layers_input = np.random.randint(1,6)
				ndense_layers_input = np.random.randint(1,5)
				

			else:
				##### WALK AROUND IN THE ALLOWABLE SPACE!
				filters_input = last_filters_input + np.random.choice([-1,0,1])
				while (filters_input < filters_bounds[0]) or (filters_input > filters_bounds[1]):
					filters_input = last_filters_input + np.random.choice([-1,0,1])

				kernel_input = last_kernel_input + np.random.choice([-1,0,1])
				while (kernel_input < kernel_bounds[0]) or (kernel_input > kernel_bounds[1]):
					kernel_input = last_kernel_input + np.random.choice([-1,0,1])

				pool_input = last_pool_input + np.random.choice([-1,0,1])
				while (pool_input < pool_bounds[0]) or (pool_input > pool_bounds[1]):
					pool_input = last_pool_input + np.random.choice([-1,0,1])

				pool_type_input = np.random.choice(['max', 'avg'])

				stride_input = last_stride_input + np.random.choice([-1,0,1])
				while (stride_input < stride_bounds[0]) or (stride_input > stride_bounds[1]):
					stride_input = last_stride_input + np.random.choice([-1,0,1])		

				dropout_input = last_dropout_input + np.random.choice([-0.05,0,0.05])
				while (dropout_input < dropout_bounds[0]) or (dropout_input > dropout_bounds[1]):
					dropout_input = last_droput_input + np.random.choice([-0.05,0,0.05])

				nconv_layers_input = last_nconv_layers_input + np.random.choice([-1,0,1])
				while (nconv_layers_input < nconv_layers_bounds[0]) or (nconv_layers_input > nconv_layers_bounds[1]):
					nconv_layers_input = last_nconv_layers_input + np.random.choice([-1,0,1])		

				ndense_layers_input = last_ndense_layers_input + np.random.choice([-1,0,1])
				while (ndense_layers_input < ndense_layers_bounds[0]) or (ndense_layers_input > ndense_layers_bounds[1]):
					ndense_layers_input = last_ndense_layers_input + np.random.choice([-1,0,1])		


			print('PREPARING TO BUILD THE MODEL!')
			time.sleep(5)
			model = createModel(nfilters=filters_input, kernel_size=kernel_input, pool_size=pool_input, pool_type=pool_type_input, strides=stride_input, dropout=dropout_input, nconv_layers=nconv_layers_input, ndense=ndense_layers_input, input_shape=input_shape, num_classes=num_classes)
			print("BUILT THE MODEL!")
			time.sleep(5)

			#### print out the structure
			model.summary()
			callbacks_list = [EarlyStopping(monitor='val_accuracy', patience=5)]

			model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
			history = model.fit(training_data, training_classes, epochs=100, validation_data=(uth_validation_data, uth_validation_classes), callbacks=callbacks_list)



			### see how it went
			"""
			plt.plot(history.history['accuracy'], label='accuracy')
			plt.plot(history.history['val_accuracy'], label='val_accuracy')
			plt.xlabel('Epoch')
			plt.ylabel('Accuracy')
			plt.legend()
			plt.show()
			""" 

			test_loss, test_acc = model.evaluate(uth_validation_data, uth_validation_classes, verbose=2)


			"""
			for pgvd, pgvc in zip(pg_validation_data, pg_validation_classes):
				reshaped_input = pgvd.reshape(1,pgvd.shape[0], pgvd.shape[1])
				prediction = model.predict_classes(reshaped_input)
				actual = pgvc
				print('Predicted / Actual: ', prediction, actual)
			"""






			#### WE WANT loss to go down, accuracy to go up!
			#if test_loss < last_test_loss:
			if test_acc > last_test_accuracy:
				#### ADOPT THE MODEL! TAKE THE STEP!
				last_filters_input = filters_input
				last_kernel_input = kernel_input
				last_pool_input = pool_input
				last_pool_type_input = pool_type_input
				last_stride_input = stride_input
				last_dropout_input = dropout_input 
				last_nconv_layers_input = nconv_layers_input
				last_ndense_layers_input = ndense_layers_input
				last_test_loss = test_loss
				last_test_accuracy = test_acc 
				improved = 'y'
				probstep = 'n'
				print("GUARANTEED STEP.")
			else:
				improved = 'n'
				#### ADOPT THE MODEL ONLY WITH SOME PROBABILITY BASED ON MODEL CHANGE
				##### THE SMALLER THE CHANGE, THE GREATER THE CHANCE WE WANT TO WALK AROUND
				#test_loss_ratio = last_test_loss / test_loss #### will be less than 1
				test_accuracy_ratio = test_acc / last_test_accuracy

				#if test_loss_ratio > np.random.random():
				if test_accuracy_ratio > np.random.random():
					#### they're close enough, take the step.
					last_filters_input = filters_input
					last_kernel_input = kernel_input
					last_pool_input = pool_input
					last_pool_type_input = pool_type_input
					last_stride_input = stride_input
					last_dropout_input = dropout_input 
					last_nconv_layers_input = nconv_layers_input
					last_ndense_layers_input = ndense_layers_input
					last_test_loss = test_loss
					last_test_accuracy = test_acc 
					probstep = 'y'
					print("PROBABILISTIC STEP.")
				else:
					probstep = 'n'
					print("REJECTED STEP.')")


			if (improved == 'y') or (probstep == 'y'):
				step += 1

				CNN_resultsfile = open(projectdir+'/Periodogram_CNN_results.csv', mode='a')
				#CNN_resultsfile.write('step,filters,kernel,pool_size,pool_type,stride,dropout,nconv_layers,ndense_layers,loss,accuracy\n')
				CNN_resultsfile.write(str(step)+','+str(filters_input)+','+str(kernel_input)+','+str(pool_input)+','+str(pool_type_input)+','+str(stride_input)+','+str(dropout_input)+','+str(nconv_layers_input)+','+str(ndense_layers_input)+','+str(test_loss)+','+str(test_acc)+'\n')
				CNN_resultsfile.close()

		except:
			traceback.print_exc()
			time.sleep(3)
			continue





except:
	traceback.print_exc()










	


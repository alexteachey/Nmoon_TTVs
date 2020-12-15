from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas
import pickle
import os
import traceback
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, AveragePooling1D, MaxPooling1D, Dropout, Dense, Flatten
from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
#from tensorflow.keras.models import Sequential 

#from keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, Dropout, Dense, Flatten
#from keras.models import Sequential
from skimage.transform import resize
import time

#tf.keras.backend.set_floatx('float64')
#tf.keras.mixed_precision.experimental.set_policy('float64')


#### LIFTED FROM HERE: https://medium.com/@mrgarg.rajat/training-on-large-datasets-that-dont-fit-in-memory-in-keras-60a974785d71
###### IN ORDER TO LOAD IN ALL THE TRAINING DATA IN BATCHES (ALL AT ONCE IS A MEMORY OVERLOAD)
class My_Custom_Generator(keras.utils.Sequence):
	def __init__(self, image_filenames, labels, batch_size) :
		self.image_filenames = image_filenames
		self.labels = labels
		self.batch_size = batch_size
    
    
	def __len__(self) :
		try:
			return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)
		except:
			(np.ceil(len(self.image_filenames) / 1.0)).astype(np.int)

  
  
	def __getitem__(self, idx) :
		batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
		batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
   	
		#return np.array([ resize( imread( filearraysdir+'/' + str(file_name) ), (80, 80, 3) ) for file_name in batch_x ]) / 255.0, np.array(batch_y)
		output_x = []
		for file_name in batch_x:
			resized_file = resize(np.load(file_name), (100,1000,1)) ### preserve the spectral channels! condense the evolution (it's interpolated anyway!)
			normed_resized_file = (resized_file - np.nanmin(resized_file)) / (np.nanmax(resized_file) - np.nanmin(resized_file))
			output_x.append(normed_resized_file)
		output_x = np.array(output_x)

		return output_x, np.array(batch_y)



#projectdir = '/data/tethys/Documents/Projects/NMoon_TTVs' #### ON TETHYS
projectdir = '/Users/hal9000/Documents/Projects/NMoon_TTVsim' #### ON HAL9000
dictdir = projectdir+'/sinusim_dictionaries'
waterfalldir = projectdir+'/sinusim_waterfalls'
filearraysdir = projectdir+'/sinusim_filearrays'


#### THIS IS THE PERIODOGRAM WATERFALL CNN PLAYGROUND
"""
Here's the question: can we use selective (sequential) removal of TTV observations to produce an evolution in the Lomb-Scargle Periodogram,
SUCH THAT generating a waterfall image (spectral evolution) can be fed into a 2D convolutional neural network to PREDICT the number of moons
in the system?!

It's not terribly well thought out / motivated right now, it's just a hunch: basically, all the moon signals are encoded in the periodogram, 
with varying strengths. Certain frequencies will dominate. We obviously cannot *add* data points to make the periodogram evolve... but we do know
that the peak power, and really all the powers, start to move around a bit in peculiar ways once you start stripping away observations. New peaks will emerge, and dominate a bit, before another peak emerges.

Now, that's just the most powerful peak. There's so much more information in the entire power spectrum. So let's look at the entire power spectrum evolve as we strip away data points, one by one. It's an 2D image, like a waterfall plot. If we can generate enough of these, it's at least conceivable that a CNN could look at these and make a determination -- is there 1,2,3,4,or 5 moons here?

The single moon case should be the easiest -- no matter how many data points you strip away, there's always just that one sinusoid there. 

What I've done in periodogram_playground.py is generate a huge number of waterfall plots, with 1-5 moons all with periods just long of 2:1 (they're 2.05:1). That way they're just a hair long of being perfect match, and you should have beating in the waveforms.

Let's see if we can get something here.
"""

"""
#### THIS ONE IS FAILING
def createModel(nfilters=4, kernel_size=(3,3), pool_size=(5,5), pool_type='avg', strides=(2,2), dropout=0.25, nconv_layers=5, ndense=4, input_shape=None, num_classes=None):
	model = Sequential()
	for convlayer in np.arange(0,nconv_layers,1):
		multiplier = 2*convlayer
		model.add(Conv2D(filters=multiplier*nfilters, kernel_size=kernel_size, padding='same', kernel_initializer='orthogonal', activation='relu', input_shape=input_shape))
		model.add(Conv2D(filters=multiplier*nfilters, kernel_size=kernel_size, padding='same', kernel_initializer='orthogonal', activation='relu'))

		if pool_type == 'avg':
			model.add(AveragePooling2D(pool_size=pool_size, strides=trides))
		elif pool_type == 'max':
			model.add(MaxPooling2D(pool_size=pool_size, strides=strides))

		model.add(Dropout(dropout))
		multiplier = 2**(convlayer+1)
		for denselayer in np.arange(0,ndense,1):
			model.add(Dense(multiplier*nfilters, activation='relu'))

		model.add(Flatten())
		model.add(Dense(num_classes, activation='sigmoid'))

		return model
"""

#### DEFINITIONS #####
#### THIS ONE IS COPIED FROM A WORKING VERSION OF THE CODE
def createModel(input_shape, num_classes, nfilters=4, kernel_size=(3,3), pool_size=(5,5), pool_type='avg', strides=(2,2), dropout=0.25, nconv_layers=5, ndense=4):
	model = Sequential()
	model.add(Conv2D(filters=nfilters, kernel_size=kernel_size, padding='same', kernel_initializer='orthogonal', activation='relu', input_shape=input_shape, data_format='channels_last')) ### 16 outputs.
	for convlayer in np.arange(0,nconv_layers,1):
		multiplier = 2**convlayer
		try:
			model.add(Conv2D(filters=multiplier*nfilters, kernel_size=kernel_size, padding='same', kernel_initializer='orthogonal', activation='relu'))
			if pool_type == 'avg':
				model.add(AveragePooling2D(pool_size=pool_size, strides=strides))
			elif pool_type == 'max':
				model.add(MaxPooling2D(pool_size=pool_size, strides=strides))	
		except:
			print('CONV2D LAYER COULD NOT BE ADDED! DIMENSIONALITY ISSUE.')
			continue

	model.add(Dropout(dropout))
	multiplier = 2**(convlayer+1)
	for denselayer in np.arange(0,ndense,1):
		model.add(Dense(multiplier*nfilters, activation='relu'))

	model.add(Flatten())
	model.add(Dense(num_classes, activation='sigmoid'))

	return model





### OK SO THIS CUBE has 10^10 elements, that would take up roughly 50 GB OF MEMORY!

### so maybe you need to figure out a better way to read this in.


try:

	nsims = len(os.listdir(dictdir))

	#data_stack = np.zeros(shape=(nsims,1000,1000))
	#class_list = np.zeros(shape=nsims)

	waterfall_filenames = []
	waterfall_classes = []


	for simnum in np.arange(0,nsims,1):
		print('loaded '+str(simnum)+' of '+str(nsims))
		waterfall_filename = waterfalldir+'/sinusim'+str(simnum)+'_waterfall.npy'
		#sim_data = np.load(waterfalldir+'/sinusim'+str(simnum)+'_waterfall.npy')
		with open(dictdir+'/sinusim'+str(simnum)+'dictionary.pkl', 'rb') as handle:
			sim_dict = pickle.load(handle)
		waterfall_class = sim_dict['Nmoons']

		waterfall_filenames.append(waterfall_filename)
		waterfall_classes.append(waterfall_class)

		### add to the input and output arrays -- CAN'T DO IT LIKE THIS!!! WAY TOO LARGE.

		#data_stack[simnum] = sim_data
		#class_list[simnum] = int(sim_dict['Nmoons'])
	#### now should have a full stack of waterfalls and classifications -- NOPE
	##### WE HAVE A LIST OF FILENAMES, AND A LIST OF THEIR CLASSIFICATIONS (IN THIS CASE, NUMBER OF MOONS)
	waterfall_filenames = np.array(waterfall_filenames)
	waterfall_classes = np.array(waterfall_classes)

	np.save(filearraysdir+'/waterfall_filenames.npy', waterfall_filenames)
	np.save(filearraysdir+'/waterfall_classes.npy', waterfall_classes)


	### separate into training, under the hood validation (uth_val), and playground validation (pg_val)

	ntraining = int(0.7*nsims)
	nuth = int(0.15*nsims)
	npg = int(0.15*nsims)

	"""
	training_data = data_stack[:ntraining]
	training_classes = class_list[:ntraining]

	uth_val_data = data_stack[ntraining:ntraining+nuth]
	uth_val_classes = class_list[ntraining:ntraining+nuth]

	pg_val_data = data_stack[ntraining+nuth:]
	pg_val_classes = data_stack[ntraining+nuth:]
	"""

	training_filenames = waterfall_filenames[:ntraining]
	training_classes = waterfall_classes[:ntraining]
	np.save(filearraysdir+'/training_waterfall_filenames.npy', waterfall_filenames)
	np.save(filearraysdir+'/training_waterfall_classes.npy', waterfall_classes)


	uth_val_filenames = waterfall_filenames[ntraining:ntraining+nuth]
	uth_val_classes = waterfall_classes[ntraining:ntraining+nuth]
	np.save(filearraysdir+'/uth_val_waterfall_filenames.npy', waterfall_filenames)
	np.save(filearraysdir+'/uth_val_waterfall_classes.npy', waterfall_classes)



	pg_val_filenames = waterfall_filenames[ntraining+nuth:]
	pg_val_classes = waterfall_classes[ntraining+nuth:]
	np.save(filearraysdir+'/pg_val_waterfall_filenames.npy', waterfall_filenames)
	np.save(filearraysdir+'/pg_val_waterfall_classes.npy', waterfall_classes)

	
	batch_size = 32
	input_shape = (100,1000,1) ### better if this is not hard coded, but whatever.

	my_training_batch_generator = My_Custom_Generator(training_filenames, training_classes, batch_size)
	my_uth_validation_batch_generator = My_Custom_Generator(uth_val_filenames, uth_val_classes, batch_size)


	#input_shape = (len(training_data[0]), 1) 
	#nput_shape = (len(training_data[0]), 1)
	#input_shape = (training_data.shape[0], training_data.shape[1], 1)
	num_classes = 6 ### 0,1,2,3,4,5


	### build the model!

	step = 0
	steps_to_take = 1000

	last_test_loss = np.inf 
	last_test_accuracy = 0 


	if os.path.exists(projectdir+'/waterfall_CNN_results.csv'):
		### read it in!
		try:
			CNN_resultsfile = pandas.read_csv(projectdir+'/waterfall_CNN_results.csv')

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
		except:
			CNN_resultsfile = open(projectdir+'/waterfall_CNN_results.csv', mode='w')
			CNN_resultsfile.write('step,filters,kernel,pool_size,pool_type,stride,dropout,nconv_layers,ndense_layers,loss,accuracy\n')
			CNN_resultsfile.close()

	else:
		#### write it new!
		CNN_resultsfile = open(projectdir+'/waterfall_CNN_results.csv', mode='w')
		CNN_resultsfile.write('step,filters,kernel,pool_size,pool_type,stride,dropout,nconv_layers,ndense_layers,loss,accuracy\n')
		CNN_resultsfile.close()

	while step < steps_to_take:
		print('step = ', step)
		try:
			#### initiate the walker

			filters_bounds = [2,4]
			kernel_bounds = [2,4]
			pool_bounds = [2,4]
			stride_bounds = [2,4]
			dropout_bounds = [0,0.4]
			nconv_layers_bounds = [1,10]
			ndense_layers_bounds = [1,10]

			if step == 0:
				#### random start!
				filters_input = np.random.randint(2,5) ### random integer between 1 and 5 (inclusive)
				kernel_input = np.random.randint(2,5) ### ditto
				kernel_input = (kernel_input, kernel_input) ### has to be a tuple for 2D!
				pool_input = np.random.randint(2,5) 
				pool_input = (pool_input, pool_input) #### pool size has to be a tuple for 2D!
				pool_type_input = np.random.choice(['max', 'avg'])
				stride_input = np.random.randint(2,5)
				stride_input = (stride_input, stride_input) ### stride has to be a tuple for 2D!
				dropout_input = np.random.choice(np.arange(0.05,0.35,0.05))
				nconv_layers_input = np.random.randint(1,5)
				ndense_layers_input = np.random.randint(1,5)
				

			else:
				##### WALK AROUND IN THE ALLOWABLE SPACE!
				filters_input = last_filters_input + np.random.choice([-1,0,1])
				while (filters_input < filters_bounds[0]) or (filters_input > filters_bounds[1]):
					filters_input = last_filters_input + np.random.choice([-1,0,1])

				#kernel_input = last_kernel_input + np.random.choice([-1,0,1]) ### 1D
				kernel_input = last_kernel_input[0] + np.random.choice([-1,0,1]) ### 2D
				kernel_input = (kernel_input, kernel_input) ### has to be a tuple for 2D!
				while (kernel_input < kernel_bounds[0]) or (kernel_input > kernel_bounds[1]):
					kernel_input = last_kernel_input[0] + np.random.choice([-1,0,1])
					kernel_input = (kernel_input, kernel_input) ### has to be a tuple for 2D!

				#pool_input = last_pool_input + np.random.choice([-1,0,1]) ### 1D
				pool_input = last_pool_input[0] + np.random.choice([-1,0,1])
				pool_input = (pool_input, pool_input) #### pool size has to be a tuple for 2D!
				while (pool_input < pool_bounds[0]) or (pool_input > pool_bounds[1]):
					pool_input = last_pool_input[0] + np.random.choice([-1,0,1])
					pool_input = (pool_input, pool_input) #### pool size has to be a tuple for 2D!

				pool_type_input = np.random.choice(['max', 'avg'])

				#stride_input = last_stride_input + np.random.choice([-1,0,1]) ### 1D
				stride_input = last_stride_input[0] + np.random.choice([-1,0,1])
				stride_input = (stride_input, stride_input) ### stride has to be a tuple for 2D!				
				while (stride_input < stride_bounds[0]) or (stride_input > stride_bounds[1]):
					stride_input = last_stride_input[0] + np.random.choice([-1,0,1])
					stride_input = (stride_input, stride_input) ### stride has to be a tuple for 2D!							

				dropout_input = last_dropout_input + np.random.choice([-0.05,0,0.05])
				while (dropout_input < dropout_bounds[0]) or (dropout_input > dropout_bounds[1]):
					dropout_input = last_droput_input + np.random.choice([-0.05,0,0.05])

				nconv_layers_input = last_nconv_layers_input + np.random.choice([-1,0,1])
				while (nconv_layers_input < nconv_layers_bounds[0]) or (nconv_layers_input > nconv_layers_bounds[1]):
					nconv_layers_input = last_nconv_layers_input + np.random.choice([-1,0,1])		

				ndense_layers_input = last_ndense_layers_input + np.random.choice([-1,0,1])
				while (ndense_layers_input < ndense_layers_bounds[0]) or (ndense_layers_input > ndense_layers_bounds[1]):
					ndense_layers_input = last_ndense_layers_input + np.random.choice([-1,0,1])		


			#raise Exception('this is as far as you want to go right now.')

			#### THIS IS THE ONE THAT'S CRAPPING OUT
			#model = createModel(nfilters=filters_input, kernel_size=kernel_input, pool_size=pool_input, pool_type=pool_type_input, strides=stride_input, dropout=dropout_input, nconv_layers=nconv_layers_input, ndense=ndense_layers_input, input_shape=input_shape, num_classes=num_classes)
			
			#### THIS IS THE ONE THAT WORKS JUST FINE FOR CONV1D
			print("PREPARING TO BUILD THE MODEL!")
			#time.sleep(5)
			model = createModel(nfilters=filters_input, kernel_size=kernel_input, pool_size=pool_input, pool_type=pool_type_input, strides=stride_input, dropout=dropout_input, nconv_layers=nconv_layers_input, ndense=ndense_layers_input, input_shape=input_shape, num_classes=num_classes)
			print('BUILT THE MODEL!')
			#time.sleep(5)

			#### print out the structure
			model.summary()


			callbacks_list = [EarlyStopping(monitor='val_accuracy', patience=5)]
			
			print("COMPILING!")
			model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
			print("COMPILED.")

			#try:
			#	model.fit_generator(generator=my_training_batch_generator, steps_per_epoch=int(3800 // int(batch_size)), epochs=100, verbose=1, validation_data=my_uth_validation_batch_generator, validation_steps=int(950 % int(batch_size)))
			#except:
			#	model.fit_generator(generator=my_training_batch_generator, steps_per_epoch=1000, epochs=100, verbose=1, validation_data=my_uth_validation_batch_generator, validation_steps=100)



			#model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
			
			print('running model.fit_generator()')
			try:
				"""
				WARNING:tensorflow:From <string>:368: Model.fit_generator (from tensorflow.python.keras.engine.training) is deprecated and will be removed in a future version.
				Instructions for updating: Please use Model.fit, which supports generators.
				"""
				#history = model.fit_generator(generator=my_training_batch_generator, steps_per_epoch=int(3800 // batch_size), epochs=100, verbose=1, callbacks=callbacks_list, validation_data=my_uth_validation_batch_generator, validation_steps=int(950 // batch_size))
				#history = model.fit(generator=my_training_batch_generator, steps_per_epoch=int(3800 // batch_size), epochs=100, verbose=1, callbacks=callbacks_list, validation_data=my_uth_validation_batch_generator, validation_steps=int(950 // batch_size))
				#history = model.fit(generator=my_training_batch_generator, epochs=100, verbose=1, callbacks=callbacks_list, validation_data=my_uth_validation_batch_generator)
				history = model.fit(x=my_training_batch_generator, epochs=100, verbose=1, callbacks=callbacks_list, validation_data=my_uth_validation_batch_generator)

			except:
				#history = model.fit_generator(generator=my_training_batch_generator, steps_per_epoch=1000, epochs=100, verbose=1, callbacks=callbacks_list, validation_data=my_uth_validation_batch_generator, validation_steps=100)
				#history = model.fit(generator=my_training_batch_generator, steps_per_epoch=1000, epochs=100, verbose=1, callbacks=callbacks_list, validation_data=my_uth_validation_batch_generator, validation_steps=100)
				history = model.fit(x=my_training_batch_generator, epochs=100, verbose=1, callbacks=callbacks_list, validation_data=my_uth_validation_batch_generator)



			### ORIGINAL BELOW, NEW VERSION ABOVE
			#history = model.fit(training_data, training_classes, epochs=100, validation_data=(uth_validation_data, uth_validation_classes), callbacks=callbacks_list)


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

		except:
			traceback.print_exc()
			time.sleep(5)



except:
	traceback.print_exc()
	#time.sleep(30)
	raise Exception('INVESTIGATE!')



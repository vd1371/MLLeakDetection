#Loading dependencies
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from multiprocessing import Queue, Process
import multiprocessing as mp
import pprint

import keras
from keras.models import Sequential, load_model
import keras.losses
from keras.layers import Dense
from keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l1, l2
from keras.models import model_from_json

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from RegressionReport import evaluate_regression

from LeakWave import h_d_measure
from Logger import Logger


import requests
import io


class DNNLeakDetector:
	
	def __init__(self, **params):

		self.layers = params.pop("layers", [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000])
		self.input_activation_func = params.pop("activation_function", "tanh")
		self.hidden_activation_func = params.pop("hidden_activation_func", "relu")
		self.final_activation_func = params.pop("final_activation_func", "linear")
		self.loss_func = params.pop("loss_func", "mse")
		self.epochs = params.pop("epochs", 500)
		self.min_delta = params.pop("min_delta", 0.00001)
		self.patience = params.pop("patience", 100)
		self.batch_size = params.pop("batch_size", 1024)
		self.should_early_stop = params.pop("should_early_stop", False)
		self.should_checkpoint = params.pop("should_checkpoint", False)
	
		self.regul_type = params.pop("regul_type", "l1")
		self.l = l1 if self.regul_type == 'l1' else l2
		self.reg_param = params.pop("reg_param", 0.000001)
		self.optimizer = params.pop("optimizer", "Adam")
		self.random_state = params.pop("random_state", 165)

		self.split_size = params.pop("split_size", 0.2)

		self.directory = "./Reports/DNN"
		self.log = Logger(address = f"{self.directory}/Log.log")

		self.input_dim = 50

	def get_data(self, batch_number):

		downloaded = False
		while not downloaded:
			x = requests.get(f"http://158.132.126.58:8000/?batch_number={batch_number}").content.decode("utf-8")

			if "NotFound" in x:
				input("The data was not found. Make sure everything is alright")
			else:
				data = pd.read_csv(io.StringIO(x), header = 0, index_col = 0)
				downloaded = True
				print (batch_number, "is downloaded from server")

		X, Y, dates = data.iloc[:, :-3], data.iloc[:, -3:], data.index 


		X_train, X_test, Y_train, Y_test, \
			dates_train, dates_test = train_test_split(X, Y, dates,
														test_size = self.split_size,
														shuffle = True,
														random_state = self.random_state)

		return X_train, X_test, Y_train, Y_test, dates_train, dates_test

	def _get_call_backs(self):
		# Creating Early Stopping function and other callbacks
		call_back_list = []
		if self.should_checkpoint:
			checkpoint = ModelCheckpoint(f'{self.directory}/BestModel.h5',
												monitor='val_loss',
												verbose=1,
												save_best_only=True,
												mode='auto')
			call_back_list.append(checkpoint)

		return call_back_list

	def log_hyperparameters(self):
		self.log.info(pprint.pformat({'layers': self.layers,
									'input_activation_func': self.input_activation_func,
									'hidden_activation_func': self.hidden_activation_func,
									'final_activation_func': self.final_activation_func,
									'loss_func': self.loss_func,
									'epochs': self.epochs,
									'min_delta': self.min_delta,
									'patience': self.patience,
									'batch_size': self.batch_size,
									'should_early_stop': self.should_early_stop,
									'regularization_type': self.regul_type,
									'reg_param': self.reg_param,
									'random_state': self.random_state}))

	def _construct_model(self):

		self.log_hyperparameters()

		model = Sequential()
		model.add(Dense(self.layers[0],
						input_dim = self.input_dim,
						activation = self.input_activation_func,
						kernel_regularizer=self.l(self.reg_param)))
		for ind in range(1,len(self.layers)):
			model.add(Dense(self.layers[ind],
							activation = self.hidden_activation_func,
							kernel_regularizer=self.l(self.reg_param)))
		model.add(Dense(3, activation = self.final_activation_func))
		 
		# Compile model
		model.compile(loss=self.loss_func,
						optimizer=self.optimizer,
						metrics = ['mape'])

		return model

	def run(self, n_rounds = 1000, warm_up = False, starting_batch = 0):

		constructed = False
		if warm_up:
			try:
				self.load_model()
				constructed = True
				self.log.info("\n\n------------\nA trained model is loaded\n------------\n\n")
			except OSError:
				print ("The model is not trained before. No saved models found")

		if not constructed:
			# Creating the structure of the neural network
			self.model = self._construct_model()
			
			# A summary of the model
			stringlist = []
			self.model.summary(print_fn=lambda x: stringlist.append(x))
			short_model_summary = "\n".join(stringlist)
			self.log.info(short_model_summary)

		call_back_list = self._get_call_backs()

		# Get data
		for batch_number in range(starting_batch, n_rounds):
			
			X_train, X_test, Y_train, Y_test, \
					dates_train, dates_test = self.get_data(batch_number)


			print ("Trying to fit to the new generated data...")
			self.model.fit(X_train.values, Y_train.values,
							validation_data=(X_test.values, Y_test.values),
							epochs=self.epochs,
							batch_size=self.batch_size,
							verbose = 2, shuffle=True, callbacks=call_back_list)

			# Evaluate the model
			train_scores = self.model.evaluate(X_train.values, Y_train.values, verbose=2)
			test_scores = self.model.evaluate(X_test, Y_test, verbose=2)
				
			print ()
			print (f'Trian_err: {train_scores}, Test_err: {test_scores}')
			self.log.info(f'batch_number:{batch_number}, Trian_err: {train_scores}, Test_err: {test_scores}')

			self.save_model()

			for i in range(3):

				y_pred_train = self.model.predict(X_train)[:, i]
				y_pred_test = self.model.predict(X_test)[:, i]

				evaluate_regression([f'OnTrain-xL{i}', X_train, Y_train, dates_train],
									[f'OnTest-xL{i}', X_test, Y_test, dates_test],
									ith_y = i,
									direc = self.directory,
									model = self.model,
									model_name = f'DNN',
									logger = self.log,
									slicer = 0.5,
									should_check_hetero = False,
									should_log_inverse = False)



			print ("----------------------------------------------------------")
		
	def load_model(self):
		
		# load json and create model
		model_type = 'BestModel' if self.should_checkpoint else 'SavedModel'
		self.model = load_model(self.directory + "/" +  f"{model_type}.h5")

	def save_model(self):
		save_address = f"{self.directory}/" 
		self.model.save(save_address + "SavedModel.h5", save_format = 'h5')


if __name__ == "__main__":

	# print (MC_sample_parallel(N=1000))

	myDetector = DNNLeakDetector(epochs = 500)
	myDetector.run(n_rounds = 1000, warm_up = False, starting_batch = 0)



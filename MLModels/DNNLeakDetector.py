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
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l1, l2
from keras.models import model_from_json

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from RegressionReport import evaluate_regression
from ClassificationReport import evaluate_classification

from LeakWave import h_d_measure
from utils import Logger
# import logging

import os
import requests
import io


class DNNLeakDetector:
	
	def __init__(self, **params):

		self.layers = params.get("layers")
		self.input_activation_func = params.get("input_activation_func")
		self.hidden_activation_func = params.get("hidden_activation_func")
		self.final_activation_func = params.get("final_activation_func")
		self.loss_func = params.get("loss_func")
		self.epochs = params.get("epochs")
		self.min_delta = params.get("min_delta")
		self.patience = params.get("patience")
		self.batch_size = params.get("batch_size")
		self.should_early_stop = params.get("should_early_stop")
		self.should_checkpoint = params.get("should_checkpoint")
	
		self.regul_type = params.get("regul_type")
		self.act_regul_type = params.get("act_regul_type")
		self.l = l2 if self.regul_type == 'l2' else l1
		self.actl = l1 if self.act_regul_type == 'l1' else l2
		self.reg_param = params.get("reg_param")
		self.dropout = params.get("dropout")
		self.optimizer = params.get("optimizer")
		self.random_state = params.get("random_state")

		self.split_size = params.get("split_size")

		self.data_directory = params.get("data_directory")
		self.directory = params.get("directory")
		self.log = Logger(address = f"{self.directory}/Log.log")
		# self.log = logging.getLogger()
		# self.log.basicConfig(filename = f"{self.directory}/Log.log", filemode = 'w')
		self.input_dim = params.get("input_dim")
		self.output_dim = params.get("output_dim")

		# self.df = df

	def get_data(self, *args, **kwargs):
		return get_data(*args, **kwargs)


	def get_data_prim(self,dataframe):
		X, Y = dataframe.iloc[:, 1:-40], dataframe.iloc[:, -40:]
		X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = self.split_size, shuffle = True, 
													random_state = self.random_state)

		scaler = StandardScaler()
		X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X_train.columns)
		X_test = pd.DataFrame(scaler.fit_transform(X_test), columns = X_test.columns)

		print (batch_number, "is read from the file")
		
		return X_train, X_test, Y_train, Y_test

	def _get_call_backs(self):
		return _get_call_backs(*args, **kwargs)

	def log_hyperparameters(self):
		return log_hyperparameters(*args, **kwargs)

	def _construct_model(self):
		return _construct_model(*args, **kwargs)

	def load_model(self):
		
		# load json and create model
		model_type = 'BestModel' if self.should_checkpoint else 'SavedModel'
		self.model = load_model(self.directory + "/" +  f"{model_type}.h5")

	def save_model(self):
		save_address = f"{self.directory}/" 
		self.model.save(save_address + "SavedModel.h5", save_format = 'h5')

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

			y_pred_train = self.model.predict(X_train)
			y_pred_test = self.model.predict(X_test)

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

	def run_prim(self, n_rounds = 1000, warm_up = False, starting_batch = 0):

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
			
			X_train, X_test, Y_train, Y_test = self.get_data_prim(batch_number)

			print ("Trying to fit to the new generated data...")
			self.model.fit(X_train, Y_train,
							validation_split=self.split_size,
							epochs=self.epochs,
							batch_size=self.batch_size,
							verbose = 2, shuffle=True,
							callbacks=call_back_list)

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

			evaluate_classification(['label', X_train, Y_train, ])


'''
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = self.split_size, shuffle = True, 
												random_state = self.random_state)

	scaler = StandardScaler()
	X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X_train.columns)
	X_test = pd.DataFrame(scaler.fit_transform(X_test), columns = X_test.columns)

	print (batch_number, "is read from the file")
	
	return X_train, X_test, Y_train, Y_test	
'''
		

if __name__ == "__main__":

	# print (MC_sample_parallel(N=1000))

	myDetector = DNNLeakDetector(epochs = 500)
	myDetector.run(n_rounds = 1000, warm_up = False, starting_batch = 0)



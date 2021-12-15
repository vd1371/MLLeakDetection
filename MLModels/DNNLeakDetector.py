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


	def get_data(self, *args, **kwargs):
		return get_data(*args, **kwargs)

	def _get_call_backs(self):
		return _get_call_backs(*args, **kwargs)

	def log_hyperparameters(self):
		return log_hyperparameters(*args, **kwargs)

	def _construct_model(self):
		return _construct_model(*args, **kwargs)

	def load_model(self):
		return _load_model(*args, **kwargs)

	def save_model(self):
		return _save_model(*args, **kwargs)

	def run(self, n_rounds = 1000, warm_up = False, starting_batch = 0):
		return run(*args, **kwargs)

if __name__ == "__main__":

	# print (MC_sample_parallel(N=1000))

	myDetector = DNNLeakDetector(epochs = 500)
	myDetector.run(n_rounds = 1000, warm_up = False, starting_batch = 0)



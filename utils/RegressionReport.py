import os
import numpy as np
import pandas as pd
import pprint
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE


# def MAPE(y_true, y_pred):
# 	y_true = np.array(y_true)
# 	y_pred = np.array(y_pred)
	
# 	if np.all(y_true):
# 		return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
# 	else:
# 		return np.nan

def R2(y_true, y_pred):
	return np.corrcoef(y_true, y_pred)[0][1]**2

def CorCoef(y_true, y_pred):
	return np.corrcoef(y_true, y_pred)[0][1]

def evaluate_regression(*args, **params):
	'''Report classification results
	
	*args should be lists of [label, x, y_true, inds, y_pred, info]

	x MUST be pandas dataframe
	y_true and y_pred must be list or 1d numpy array
	'''
	report_directory = params.get('report_directory')
	# model = params.get('model')
	model_name = params.get('model_name')
	logger = params.get('logger')
	# should_check_hetero = params.get('should_check_hetero', False)
	# should_log_inverse = params.get('should_log_inverse', False)
	# ith_y = params.get("ith_y", 0)

	for ls in args:

		label, x, y_true, inds, y_pred, info = ls
		# y_pred = model.predict(x)

		#For the output of the DNN regression
		# if len(np.shape(y_pred)) > 1:
		# 	y_pred = y_pred[:, ith_y]

		# if model_name == "DNN":
		# 	y_true = y_true.values[:, ith_y]
		
		# Saving into csv file
		if not model_name == "DNN":
			report = pd.DataFrame()
			report['Actual'] = y_true
			report['Predicted'] = y_pred
			report['Error'] = report['Actual'] - report['Predicted']
			report['Ind'] = inds
			for col in info.columns:
				report[col] = info[col].values
			report.set_index('Ind', inplace=True)
			report.to_csv(report_directory + "/" + f'{model_name}-{label}.csv')


		corcoef_ = CorCoef(y_true, y_pred)
		r2_ = R2(y_true, y_pred)
		mse_ = MSE(y_true, y_pred)
		mae_ = MAE(y_true, y_pred)
		# mape_ = MAPE(y_true, list(y_pred))
		
		# Reporting the quantitative results
		report_str = f"{label}, CorCoef= {corcoef_:.4f}, "\
						f"R2= {r2_:.4f}, RMSE={mse_**0.5:.4f}, "\
							f"MSE={mse_:.4f}, MAE={mae_:.4f}, "
		
		logger.info(report_str)
		print(report_str)













		# # Comparing the results of all models
		# compare_direc = os.path.abspath(direc + "/../") + "/Comparision.csv"
		# if os.path.exists(compare_direc):
		# 	compar_df = pd.read_csv(compare_direc, index_col = 0)
		# else:
		# 	compar_df = pd.DataFrame(columns = ['CorrCoef', 'R2', 'MSE', 'MAE', 'MAPE'])

		# compar_df.loc[model_name + "-" + label] = [corcoef_, r2_, mse_, mae_, mape_]
		# compar_df.to_csv(compare_direc)

		
		# Plotting errors
		# errs = list(report['Error'])
		# ticks = [i for i in range(len(errs))]
		# plt.clf()
		# plt.ylabel('Erros')
		# plt.title(label+'-Errors')
		# plt.scatter(ticks, errs, s = 1)
		# plt.grid(True)
		# plt.savefig(direc + '/' + label + '-Errors.png')
		# plt.close()

		# if slicer != 1:
		# 	y_true, y_pred = y_true[-int(slicer*len(y_true)):], y_pred[-int(slicer*len(y_pred)):]
		
		# # let's order them
		# temp_list = []
		# for true, pred in zip(y_true, y_pred):
		# 	temp_list.append([true, pred])
		# temp_list = sorted(temp_list , key=lambda x: x[0])
		# y_true, y_pred = [], []
		# for i, pair in enumerate(temp_list):
		# 	y_true.append(pair[0])
		# 	y_pred.append(pair[1])
			
		# # Actual vs Predicted plotting
		# plt.clf()
		# plt.xlabel('Actual')
		# plt.ylabel('Predicted')
		# plt.title(label+'-Actual vs. Predicted')
		# ac_vs_pre = plt.scatter(y_true, y_pred, s = 2)
		# plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'k--', lw=0.75)
		# plt.grid(True)
		# plt.savefig(direc + '/' + label + '-ACvsPRE.png')
		# plt.close()
		
		# # Actual and Predicted Plotting
		# plt.clf()
		# plt.xlabel('Sample')
		# plt.ylabel('Value')
		# plt.title(label + "-Actual and Predicted")
		# ticks = [i for i in range(len(y_true))]
		# act = plt.plot(ticks, y_true, label = "actual")
		# pred = plt.plot (ticks, y_pred, label = 'predicted')
		# plt.legend()
		# plt.grid(True)
		# plt.savefig(direc + '/' + label + '-ACandPRE.png')
		# plt.clf()
		# plt.close()

		# # Plotting the error percentage vs. feature and the output variable
		# if should_check_hetero:
		# 	try:

		# 		if np.all(y_true):
		# 			error_vec = (np.array(y_pred) - np.array(y_true)) / np.array(y_true)
		# 		else:
		# 			error_vec = np.abs(np.array(y_pred) - np.array(y_true))

		# 		plt.clf()
		# 		plt.scatter(y_true, error_vec, s = 1)
		# 		plt.savefig(f"{direc}/{label}-Hetero")

		# 		# To add the Y column for the hetero analysis
		# 		data = x.copy()
		# 		data['Y'] = y_true

		# 		file_counter = 0
		# 		first = True
		# 		for i, col in enumerate(data.columns):

		# 			# Creating the figs and axes
		# 			if first:
		# 				fig, ax = plt.subplots(nrows=3, ncols=3)
		# 				fig.tight_layout()
		# 				first = False

		# 			counter = i % 9
		# 			row_idx = int (counter/3)
		# 			col_idx = counter % 3

		# 			ax[row_idx, col_idx].set_title(col)
		# 			ax[row_idx, col_idx].scatter(data[col], error_vec, s = 1)

		# 			if (i % 9 == 8) or (i == len(data.columns)-1):
						
		# 				# Unless for the first time, files shoud be saved
		# 				plt.savefig(f"{direc}/{label}-Hetero-{file_counter}")
		# 				plt.close()

		# 				file_counter += 1

		# 				if i != len(data.columns)-1:
		# 					fig, ax = plt.subplots(nrows=3, ncols=3)
		# 					fig.tight_layout()
							
		# 	except ZeroDivisionError:
		# 		print ("Unable to plot heteroskedasticity graphs. Output variable contains zero")

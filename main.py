######################################################################################################################
# MAIN: Train the ML models for one noise variance and then validate the model with the same variance.
######################################################################################################################

import os
import numpy as np

# for sorting lists naturally
import natsort

import pandas as pd

import matplotlib.pyplot as plt

import math

######################################################################################################################
# Import the modules created by us
######################################################################################################################
# to get the data
from get_data import HandleData

# to save the results
from save_logs import logs

# to handle Machine Learning model
from model import *

######################################################################################################################
# import sklearn modules
######################################################################################################################
# Split arrays or matrices into random train and test subsets
from sklearn.model_selection import train_test_split

# Machine Learning models
from sklearn.linear_model import LinearRegression,  RidgeClassifier, SGDClassifier , SGDRegressor, LogisticRegression #SGDClassifier bad
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, BaggingRegressor, RandomForestRegressor, VotingClassifier, VotingRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR          #SVR NO, DEMORA MUCHISIMO
from sklearn.neural_network import MLPRegressor

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
import time

def saveValidationLabels(label_validation, path):
	# Create a folder folder_save
	objectLogs.createFolder(path)

	# Clean the file if it already exists
	open(path + 'azimuthToValidation' + str(SNR) + '.csv', 'w').close()
	open(path + 'elevationToValidation' + str(SNR) + '.csv', 'w').close()
	
	azimuth_validation = label_validation.loc[:,'azimuth']
	elevation_validation = label_validation.loc[:,'elevation']
	
	objectLogs.writeFile(path + 'azimuthToValidation' + str(SNR) + '.csv', azimuth_validation)
	objectLogs.writeFile(path + 'elevationToValidation' + str(SNR) + '.csv', elevation_validation)


if __name__ == '__main__':
	#Type of antennas in the receiving system. Each type of antenna has its dataset
	antenna_type = 'dipoleVee'
	
	experiment = 'RegressionCorr'

	data = 'data_corr_antenna_plus'
	# Folder where the dataset is for the receiving antennas defined in the variable antenna_type
	data_folder = './' + antenna_type + '/data/' + data
	# Folder where all results of the ML models for the dataset are saved
	logs_folder = './' + antenna_type + '/results/' + experiment + '/'

	label_name = ['azimuth','elevation']

	# Instance of the class used to save the results of the ML models
	objectLogs = logs()
	
	# Percentage of the dataset that will be used to validate the ML models
	test_percentage = 0.2

	# Seed for same random results
	random_state = 42

	# Resolution of the proposal. If the model is going to solve the angles with a difference of 1 degrees, 2 degrees...
	# This variable will be used to obtain data from the dataset
	phase_list = [1]
	
	# List containing the ML models to be used

	# dt = DecisionTreeRegressor(criterion='friedman_mse', max_depth=562
 # , min_samples_split=25, min_samples_leaf=23, splitter='best', max_features='auto', random_state=random_state)
	# dt1 = DecisionTreeRegressor(random_state=random_state)
	# ML_model_dict = {'VotingRegressor': MultiOutputRegressor(VotingRegressor(estimators=[('dtOptuna', dt), ('dt', dt1)]))}
	
	# ML_model_dict = {'DecisionTreeRegressor': DecisionTreeRegressor(criterion='friedman_mse', max_depth=562
	#  	, min_samples_split=25, min_samples_leaf=23, splitter='best', max_features='auto', random_state=random_state)}

	#ML_model_dict = {'DecisionTreeRegressor': DecisionTreeRegressor(random_state=random_state)}

	# ML_model_dict = {'DecisionTreeRegressor': DecisionTreeRegressor(criterion='friedman_mse', max_depth=666
	#  	, min_samples_split=18, min_samples_leaf=15, splitter='random', max_features='sqrt', random_state=random_state)}

	#ML_model_dict = {'MLPRegressor': MultiOutputRegressor(MLPRegressor())}
	ML_model_dict = {'RandomForestRegressor': RandomForestRegressor(n_estimators=10)}

	# dt = DecisionTreeRegressor(criterion='friedman_mse', max_depth=666
 # , min_samples_split=18, min_samples_leaf=15, splitter='random', max_features='sqrt', random_state=random_state)
	# dt1 = DecisionTreeRegressor(random_state=random_state)
	# ML_model_dict = {'VotingRegressor': MultiOutputRegressor(VotingRegressor(estimators=[('dtOptuna', dt), ('dt', dt1)]))}

	min_distance = 123
	distanceList = [213]
	
	list_proposal = [Multioutput]

	color_list = ['b', 'y', 'g', 'r', 'm', 'c', 'k']

	label_name_validation = [i + ' validation' for i in label_name]
	label_name_predict = [i + ' predict' for i in label_name]
	
	label_name_mse = [i + ' mse' for i in label_name]
	label_name_accuracy = [i + ' accuracy' for i in label_name]


	for phase in phase_list:
		print('\nphase angle: ' + str(phase))

		# For each ML model defined in ML_model_dict
		for ML_model_key in ML_model_dict:
			ML_model = ML_model_dict[ML_model_key]
			print('\n'+ ML_model_key + ' :' , ML_model)

			color_index = 0

			# Obtain the results of the proposal for different number of receiving antennas
			#for antenna_number in natsort.natsorted(os.listdir(data_folder)):
			for antenna_number in ['4','6','8','10','12','14','16']:
	
				print('\nnumber of antennas: ' + str(antenna_number))
				
				# Create the dataframes that will be saved
				labels = pd.DataFrame(columns=['antenna number', 'SNR', 'color graphs'] + label_name_validation + label_name_predict)
				logs_df = pd.DataFrame(columns=['antenna number', 'SNR', 'prediction time', 'accuracy', 'mse'] + label_name_accuracy + label_name_mse)
				
				color_antenna = color_list[color_index]
				color_index += 1

				#################################################################################################################################################################################
				################################## Get the dataset for 'antenna_number' number of antennas ######################################################################################
				handleData = HandleData(phase, folder=data_folder + '/' + antenna_number, until='all')
				
				print('\nGetting data...')
				dataset = handleData.get_synthatic_data()
				#################################################################################################################################################################################
						
				for proposal in list_proposal:
					print(proposal.__name__)

					#################################################################################################################################################################################
					################################## Preparation of the files where the results of the proposal will be saved #####################################################################
					# Folder where the results of the proposal for each resolution and each number of antennas are saved
					folder_save = logs_folder + str(phase) + '/' + proposal.__name__ + '/' + str(antenna_number) + '/'
			
					# Create a folder folder_save
					objectLogs.createFolder(folder_save)

					# # Create a folder folder_save_comparison
					# objectLogs.createFolder(folder_save_comparison)

					# Clean the file if it already exists: folder_save + ML_model_key + '.csv'
					open(folder_save + ML_model_key + '.csv', 'w').close()

					# Clean the file if it already exists: folder_save + ML_model_key + 'label_validation.csv''
					open(folder_save + ML_model_key + '_labels.csv', 'w').close()

					#################################################################################################################################################################################
					

					#################################################################################################################################################################################
					################################## MAIN #################################################################################################################################
					# Get the dataset to each SNR
					dataset_by_snr = dataset.groupby('SNR')
					
					
					for snr_by_snr in dataset_by_snr:
						SNR = snr_by_snr[0]
						
						# Get the power values obtained in the receiving antennas to train the ML model.
						power_data = snr_by_snr[1].loc[:, 'Pr0':]
						power_data = power_data.iloc[:, :-1]

						# Normalize power_data
						power_data = power_data.div(power_data.sum(axis=1), axis=0)

						label_data = snr_by_snr[1].loc[:, :label_name[-1]]

						power_data_train, power_data_validation, label_train, label_validation = train_test_split(power_data, label_data, test_size=test_percentage, random_state=random_state)
						#saveValidationLabels(label_validation, './' + antenna_type + '/validation_angles/' + str(phase) + '/' + proposal.__name__ + '/' + str(antenna_number) + '/')

						clf = ML_model
						clf.fit(power_data_train, label_train)
						
						start_time = time.time()
						y_pred = clf.predict(power_data_validation)
						time_pred = time.time() - start_time
						
						y_pred = pd.DataFrame(y_pred, columns = label_name_predict)
			
						antenna_number_tem = [int(antenna_number) for i in range(y_pred.shape[0])]
	
						SNR_tem = [int(SNR) for i in range(y_pred.shape[0])]

						accuracy_per_target = []
						mean_squared_error_per_target = []
						for i in range(y_pred.shape[1]):
							label_pred_target = y_pred.iloc[:,i]
							label_validation_target = label_validation.iloc[:,i]	
							mean_squared_error_per_target.append(mean_squared_error(label_validation_target, label_pred_target))
							label_pred_target = np.round(y_pred.iloc[:,i], decimals=0)
							accuracy_per_target.append(accuracy_score(label_validation_target, label_pred_target))

						mse = sum(mean_squared_error_per_target)
						accuracyOfModel = np.prod(accuracy_per_target)

						listResults = [antenna_number, SNR, time_pred, accuracyOfModel, mse] + accuracy_per_target + mean_squared_error_per_target

						# df.loc() function to add a row to the end of a pandas
						logs_df.loc[len(logs_df.index)] = listResults
						print(logs_df)

						# This is important to make concat then
						label_validation = label_validation.reset_index(drop=True)

						#label_validation = label_validation.rename(columns={'azimuth': 'azimuth_validation', 'elevation': 'elevation_validation'})
						label_validation.columns = label_name_validation

						# Join label_validation and y_pred dataframes
						label_temp = pd.concat([label_validation, y_pred], axis = 1, join='inner')

						label_temp.insert(0,'color graphs', color_antenna)
						label_temp.insert(0,'SNR', SNR_tem )
						label_temp.insert(0,'antenna number', antenna_number_tem)

						labels = labels.append(label_temp, ignore_index=True)
			
				labels.to_csv(folder_save + ML_model_key + '_labels.csv', mode='w')
				logs_df.to_csv(folder_save + ML_model_key + '.csv', mode='w')
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
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score

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
	
	experiment = 'VotingRegressorCov'

	# Folder where the dataset is for the receiving antennas defined in the variable antenna_type
	data_folder = './' + antenna_type + '/data_with_loss_cov'
	# Folder where all results of the ML models for the dataset are saved
	logs_folder = './' + antenna_type + '/results/' + experiment + '/'

	# # Folder where all results of the ML models for the dataset are saved
	# logs_folder_comparison = './' + antenna_type + '/results_comparison/' + experiment + '/' + 'Scatter graphs' + '/'

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

	#ML_model_dict = {'DecisionTreeRegressor': DecisionTreeRegressor(random_state=random_state)}

	dt = DecisionTreeRegressor(criterion='friedman_mse', max_depth=666
 , min_samples_split=18, min_samples_leaf=15, splitter='random', max_features='sqrt', random_state=random_state)
	dt1 = DecisionTreeRegressor(random_state=random_state)
	ML_model_dict = {'VotingRegressor': MultiOutputRegressor(VotingRegressor(estimators=[('dtOptuna', dt), ('dt', dt1)]))}

	min_distance = 123
	distanceList = [213]
	
	list_proposal = [Multioutput]

	color_list = ['b', 'y', 'g', 'r', 'm', 'c', 'k']


	for phase in phase_list:
		print('\nphase angle: ' + str(phase))

		# For each ML model defined in ML_model_dict
		for ML_model_key in ML_model_dict:
			ML_model = ML_model_dict[ML_model_key]

			color_index = 0

			# Obtain the results of the proposal for different number of receiving antennas
			#for antenna_number in natsort.natsorted(os.listdir(data_folder)):
			print('Cov')
			for antenna_number in ['4','6','8','10','12']:
	
				print('\nnumber of antennas: ' + str(antenna_number))
				
				# Create the dataframes that will be saved
				labels = pd.DataFrame(columns=['antenna_number', 'SNR', 'color_graphs_3D', 'azimuth_validation', 'azimuth_predict', 'elevation_validation','elevation_predict'])
				logs_df = pd.DataFrame(columns=['antenna_number', 'SNR', 'accuracyOfModel', 'mse', 'accuracy_per_target', 'mean_squared_error_per_target'])
				
				color_antenna = color_list[color_index]
				color_index += 1

				#################################################################################################################################################################################
				################################## Get the dataset for 'antenna_number' number of antennas ######################################################################################
				handleData = HandleData(phase, folder=data_folder + '/' + antenna_number, until='all')
				
				print('\nGetting data...')
				dataset = handleData.get_synthatic_data()
				#################################################################################################################################################################################
						
				power_data = dataset.loc[:, 'Pr0':]
				print(power_data)
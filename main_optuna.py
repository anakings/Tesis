######################################################################################################################
# MAIN: Train the ML models for one noise variance and then validate the model with the same variance.
######################################################################################################################

import os
import numpy as np

# for sorting lists naturally
import natsort

######################################################################################################################
# Import the modules created by us
######################################################################################################################
# to get the data
from get_data import HandleData

# to save the results
from save_logs import logs

# to handle Machine Learning model
from model import *

# to Hyperparameter Optimization Analysis
# from optuna_setup import Objective
import optuna_setup, optuna_setup_classifier

######################################################################################################################
# import sklearn modules
######################################################################################################################
# Split arrays or matrices into random train and test subsets
from sklearn.model_selection import train_test_split

# Machine Learning models
from sklearn.linear_model import LinearRegression,  RidgeClassifier, SGDClassifier, SGDRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
# from sklearn.model_selection import GridSearchCV
import optuna

def saveLabels(label_validation, path):
	# Create a folder folder_save
	objectLogs.createFolder(path)

	# Clean the file if it already exists
	open(path + 'azimuthLabel' + str(SNR) + '.csv', 'w').close()
	open(path + 'elevationLabel' + str(SNR) + '.csv', 'w').close()
	
	azimuth_validation = label_validation.loc[:,'azimuth']
	elevation_validation = label_validation.loc[:,'elevation']
	
	objectLogs.writeFile(path + 'azimuthLabel' + str(SNR) + '.csv', azimuth_validation)
	objectLogs.writeFile(path + 'elevationLabel' + str(SNR) + '.csv', elevation_validation)


if __name__ == '__main__':
	#Type of antennas in the receiving system. Each type of antenna has its dataset
	antenna_type = 'dipoleVee'
	
	experiment = 'optuna_cov_class'

	# Folder where the dataset is for the receiving antennas defined in the variable antenna_type
	data_folder = './' + antenna_type + '/data_with_loss_cov'
	# Folder where all results of the ML models for the dataset are saved
	logs_folder = './' + antenna_type + '/results/' + experiment + '/'

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
	ML_model_dict = {'DT_Regressor' : DecisionTreeRegressor(random_state=random_state)}

	min_distance = 123
	distanceList = [213]
	
	list_proposal = [Multioutput]

	for phase in phase_list:
		print('\nphase angle: ' + str(phase))

		# For each ML model defined in ML_model_dict
		for ML_model_key in ML_model_dict:
			ML_model = ML_model_dict[ML_model_key]
			print('\n'+ ML_model_key + ' :' , ML_model)

			# Obtain the results of the proposal for different number of receiving antennas
			for antenna_number in natsort.natsorted(os.listdir(data_folder)):
			#for antenna_number in ['12','14','16']:
				print('\nnumber of antennas: ' + str(antenna_number))

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

					# Clean the file if it already exists: folder_save + ML_model_key + '.csv'
					open(folder_save + ML_model_key + '.csv', 'w').close()
					#################################################################################################################################################################################
					

					#################################################################################################################################################################################
					################################## MAIN #################################################################################################################################
					# Get the dataset to each SNR
					dataset_by_snr = dataset.groupby('SNR')
					for snr_by_snr in dataset_by_snr:
						SNR = snr_by_snr[0]
						print('\nSNR:', f'{SNR}dB')
						
						# Get the power values obtained in the receiving antennas to train the ML model.
						power_data = snr_by_snr[1].loc[:, 'Pr0':]

						# Normalize power_data
						power_data = power_data.div(power_data.sum(axis=1), axis=0)

						# Get the labels to train the ML model. In this case the labels will be made up of the azimuth and elevation angles
						label_data = snr_by_snr[1].loc[:, :'elevation']
			
						x_train, x_val, y_train, y_val = train_test_split(power_data, label_data, random_state=random_state)

						saveLabels(y_train, './' + antenna_type + '/training_angles/' + str(phase) + '/' + proposal.__name__ + '/' + str(antenna_number) + '/')
						saveLabels(y_val, './' + antenna_type + '/validation_angles/' + str(phase) + '/' + proposal.__name__ + '/' + str(antenna_number) + '/')

						mse, best_params, best_trial = optuna_setup_classifier.optuna_main(x_train, x_val, y_train, y_val)

						listValues = [antenna_number, SNR, mse, best_params, best_trial]
						print(listValues)

						objectLogs.writeFile(folder_save + ML_model_key + '.csv', listValues)

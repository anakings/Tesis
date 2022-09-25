######################################################################################################################
# MAIN: Train the ML models for one noise variance and then validate the model with the same variance.
######################################################################################################################

import os

# for sorting lists naturally
import natsort

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

######################################################################################################################
# Import the modules created by us
######################################################################################################################
# to get the data
from get_data import HandleData



if __name__ == '__main__':
	#Type of antennas in the receiving system. Each type of antenna has its dataset
	antenna_type = 'dipoleVee'

	data = 'data_power_with_loss'
	
	# Folder where the dataset is for the receiving antennas defined in the variable antenna_type
	data_folder = './' + antenna_type + '/data/' + data
	# Folder where all results of the ML models for the dataset are saved
	logs_folder = './' + antenna_type + '/data analytics' + '/' + data + '/'

	# Resolution of the proposal. If the model is going to solve the angles with a difference of 1 degrees, 2 degrees...
	# This variable will be used to obtain data from the dataset
	phase_list = [1]

	color_list = ['b', 'y', 'g', 'r', 'm', 'c', 'k']


	for phase in phase_list:
		print('\nphase angle: ' + str(phase))

		color_index = 0

		# Obtain the results of the proposal for different number of receiving antennas
		for antenna_number in natsort.natsorted(os.listdir(data_folder)):
		#for antenna_number in ['4','']:
	
			print('\nnumber of antennas: ' + str(antenna_number))
				
			color_antenna = color_list[color_index]
			color_index += 1

			#################################################################################################################################################################################
			################################## Get the dataset for 'antenna_number' number of antennas ######################################################################################
			handleData = HandleData(phase, folder=data_folder + '/' + antenna_number, until='all')
				
			print('\nGetting data...')
			dataset = handleData.get_synthatic_data()
			
			#################################################################################################################################################################################
			################################## Preparation of the files where the results of the proposal will be saved #####################################################################
			# Folder where the results of the proposal for each resolution and each number of antennas are saved
			folder_save = logs_folder + str(antenna_number) + '/'
			if not os.path.exists(folder_save):
				os.makedirs(folder_save)
			#################################################################################################################################################################################		

			# Get the dataset to each SNR
			dataset_by_snr = dataset.groupby('SNR')	
					
			for snr_by_snr in dataset_by_snr:
				SNR = snr_by_snr[0]
					
				# Get the power values obtained in the receiving antennas to train the ML model.
				power_data = snr_by_snr[1].loc[:, 'Pr0':]

				plt.clf() 
				plt.cla()

				# for count, value in enumerate(power_data.columns):
				# 	axis_x = [count for j in power_data.index]
				# 	axis_y = power_data.loc[:, value]
				# 	axis_y = axis_y.to_numpy()
				# 	plt.plot(axis_x, axis_y, color=color_antenna)

				# plt.xlabel('elements covariance')
				# plt.ylabel('values')

				# plt.savefig(folder_save + 'SNR_' + str(SNR) + '.png')
				

				# Plot data
				axis_x = np.ones((power_data.shape[0],power_data.shape[1]))
				for i in range(1,power_data.shape[1]+1):
					axis_x[:,i-1] = i
				plt.plot(axis_x, power_data, color=color_antenna)
				plt.xlabel('elements covariance')
				plt.ylabel('values')
				plt.savefig(folder_save + 'SNR_' + str(SNR) + '.png')
			# 	X = power_data
			# 	distributions = [
   #  			("Unscaled data", X),
   #  			("Data after standard scaling", StandardScaler().fit_transform(X)),
   #  			("Data after min-max scaling", MinMaxScaler().fit_transform(X)),
   #  			("Data after max-abs scaling", MaxAbsScaler().fit_transform(X)),
   #  			(
   #  			    "Data after robust scaling",
   #  			    RobustScaler(quantile_range=(25, 75)).fit_transform(X),
   #  			),
   #  			(
   #  			    "Data after power transformation (Yeo-Johnson)",
   #  			    PowerTransformer(method="yeo-johnson").fit_transform(X),
   #  			),
   #  			(
   #  			    "Data after power transformation (Box-Cox)",
   #  			    PowerTransformer(method="box-cox").fit_transform(X),
   #  			),
   #  			(
   #  			    "Data after quantile transformation (uniform pdf)",
   #  			    QuantileTransformer(output_distribution="uniform").fit_transform(X),
   #  			),
   #  			(
   #  			    "Data after quantile transformation (gaussian pdf)",
   #  			    QuantileTransformer(output_distribution="normal").fit_transform(X),
   #  			),
   #  			("Data after sample-wise L2 normalizing", Normalizer().fit_transform(X)),
			# 
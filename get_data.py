import os
import natsort
from scipy import io
import numpy as np
import pandas as pd

class HandleData():
	def __init__(self, phase, folder, until='all'):
		self.folder = folder

		# granularity based on angles
		self.phase = phase

		# up to which file you want to select (up to what distance you want to select)
		self.until = None if until == 'all' else until

	def get_synthatic_data(self):
		# sort files until [until]
		list_files = natsort.natsorted(os.listdir(self.folder))[0:self.until]

		list_files_len = len(list_files) if self.until == None else self.until
		print('Number of files used for training:', list_files_len)

		# select files
		list_index = [x+1 for x in range(0, list_files_len)]

		dataset_list = []

		for index in list_index:
			for file in (list_files[index-1:index]):
				# get power, distance, quadrant, and snr matrix data
				content = io.loadmat(self.folder + '/' + file)
				matrix_name = list(content.keys())[-1]
				org_data = content[matrix_name]
				org_data = org_data[::self.phase,::self.phase,:] # granularity based on angles: get data every [phase] angles

				az_angles = org_data.shape[0] # total number of azimuth angles
				el_angles = org_data.shape[1] # total number of elevation angles
				pr_d_q_snr = org_data.shape[2] # number of antennas + distance + quadrant + snr
				snr_total = org_data.shape[3] # total number of different snr

				antenna_total = pr_d_q_snr - 3 # total number of antennas

				# [label_data_temp] contains the values of the azimuth and elevation angles with shape (az_angles, el_angles, 2)
				label_data_temp = np.zeros(shape=(az_angles, el_angles, 2))
				for j in range(label_data_temp.shape[0]):
					for k in range(label_data_temp.shape[1]):
						label_data_temp[j,k] = np.array([j,k])
				
				# [label_array] is reshaped as (az_angles * el_angles * snr_total, 2): two columns, one for azimuth and one for elevation
				self.label_array = np.concatenate(np.array([label_data_temp for i in range(snr_total)]), axis=0)
				self.label_array = self.label_array.reshape(self.label_array.shape[0]*self.label_array.shape[1], 2)

				# data for every feature
				power_list = []
				distance_list = []
				quadrant_list = []
				snr_list = []

				# get data for each snr
				for snr_index in range(snr_total):
					# get power data by specific snr (az_angles, el_angles, antenna_total)
					power_by_snr = org_data[:, :, :antenna_total, snr_index]
					# reshape as (az_angles * el_angles, antenna_total)
					power_by_snr_reshaped = power_by_snr.reshape(az_angles * el_angles, antenna_total)
					
					# get distance data by specific snr (az_angles, el_angles, 1)
					distance_by_snr = org_data[:, :, antenna_total, snr_index]
					# reshape as (az_angles * el_angles, 1)
					distance_by_snr_reshaped = distance_by_snr.reshape(az_angles * el_angles, 1)
					
					# get quadrant data by specific snr (az_angles, el_angles, 1)
					quadrant_by_snr = org_data[:, :, antenna_total + 1, snr_index]
					# reshape as (az_angles * el_angles, 1)
					quadrant_by_snr_reshaped = quadrant_by_snr.reshape(az_angles * el_angles, 1)

					# get snr data by specific snr (az_angles, el_angles, 1)
					snr_by_snr = org_data[:, :, antenna_total + 2, snr_index]
					# reshape as (az_angles * el_angles, 1)
					snr_by_snr_reshaped = snr_by_snr.reshape(az_angles * el_angles, 1)

					power_list.append(power_by_snr_reshaped)
					distance_list.append(distance_by_snr_reshaped)
					quadrant_list.append(quadrant_by_snr_reshaped)
					snr_list.append(snr_by_snr_reshaped)

				power = np.concatenate(tuple([i for i in power_list]), axis=0)
				distance = np.concatenate(tuple([i for i in distance_list]), axis=0)
				quadrant = np.concatenate(tuple([i for i in quadrant_list]), axis=0)
				snr = np.concatenate(tuple([i for i in snr_list]), axis=0)
				
				dataset = np.concatenate((self.label_array, distance, quadrant, snr, power), axis=1)
				dataset_list.append(dataset)

		# legend: column label of dataframe
		column_power = ['Pr' + str(i) for i in range(antenna_total)]
		column_label = ['azimuth', 'elevation', 'distance', 'quadrant', 'SNR']
		column_list = column_label + column_power

		# create a dataframe with labels
		dataset = np.concatenate(tuple([i for i in dataset_list]), axis=0)
		datasetPd = pd.DataFrame(dataset, columns=column_list)
		print(datasetPd)

		return datasetPd
		
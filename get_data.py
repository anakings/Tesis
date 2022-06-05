import os
import natsort
from scipy import io
import numpy as np
import pandas as pd

class HandleData():                                  
	def __init__(self, phase, folder, phase_distance='all', until='all', samples_by_distance='all'):
		self.folder = folder

		# granularity based on angles
		self.phase = phase

		# point from which file you want to take the data giving jumps equal to the specified number
		self.phase_distance = None if phase_distance == 'all' else phase_distance

		# up to which file you want to select (up to what distance you want to select)
		self.until = None if until == 'all' else until

		# [samples_by_distance] is how many files of the same distance you want to take, 
		# how many files are to be taken from the file pointed to by the phase_distance pointer.
		self.samples_by_distance = None if samples_by_distance == 'all' else samples_by_distance

	def get_synthatic_data(self):
		# sort files until [until]
		list_files = natsort.natsorted(os.listdir(self.folder))[0:self.until]

		# select files until [until] every [phase_distance]
		list_index = [x+1 for x in range(0, self.until, self.phase_distance)]

		for index in list_index:
			for file in (list_files[index-1:index+self.samples_by_distance-1]):
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

				# legend: column label of dataframe
				column_power = ['Pr'+ str(i) for i in range(power.shape[1])]
				column_label = ['azimuth', 'elevation', 'distance', 'quadrant', 'SNR']
				column_list = column_label + column_power
				
				dataset = np.concatenate((self.label_array, distance, quadrant, snr, power), axis=1)

				# create a dataframe with labels
				dataset = pd.DataFrame(dataset, columns=column_list)
				print(dataset)

		return dataset
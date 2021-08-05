from scipy import io
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder

class HandleData():                                  

	def __init__(self, phase, folder, one_hot_encode=False, iterations='all'):

		self.folder = folder
		content = io.loadmat(folder + '/' + os.listdir(folder)[0])
		matrix_name = list(content.keys())[-1]
		data = content[matrix_name]
		self.phase = phase
		self.data = data[::self.phase,::self.phase,:]
		self.data_set = np.empty((0, self.data.shape[2]), int)

		if self.data.shape[1] != 1:
			self.label_array = np.empty((0, 2), int)
		else:
			self.label_array = np.empty((0, 1), int)
		
		self.one_hot_encode = one_hot_encode

		if iterations == 'all':
			self.iterations = None
		else:
			self.iterations = iterations

		#print(self.data.shape)
		#print(self.data_set.shape)
		if self.one_hot_encode:
			# self.label_set_one_hot_encoded = np.empty((0, self.data.shape[0]), dtype=np.float32)
			self.label_set_one_hot_encoded = np.zeros((len(os.listdir(folder)[:self.iterations])*self.data.shape[0], self.data.shape[0]), dtype=np.float32)
		else:
			self.label_list = []

	def one_hot_encode(self, index):
		encoded_no = np.zeros(self.data.shape[0], dtype=np.float32)
		encoded_no[index] = 1
		return encoded_no

	def get_synthatic_data(self):

		for index, file in enumerate(os.listdir(self.folder)[:self.iterations]):
	
			content = io.loadmat(self.folder + '/' + file)
			matrix_name = list(content.keys())[-1]
			data = content[matrix_name]
			data = data[::self.phase,::self.phase,:]

			label_data_temp = np.zeros(shape=(data.shape[0], data.shape[1], 2))
	
			data_reshaped = data.reshape(data.shape[0]*data.shape[1], data.shape[2])
	
			self.data_set = np.vstack((self.data_set, data_reshaped))

			for j in range(label_data_temp.shape[0]):
				for k in range(label_data_temp.shape[1]):
					label_data_temp[j,k] = np.array([j,k])
					if self.one_hot_encode:
						# self.label_set_one_hot_encoded = np.vstack((self.label_set_one_hot_encoded, self.one_hot_encode(j*label_data_temp.shape[1]+k)))
						self.label_set_one_hot_encoded[index*self.data.shape[0]+j, j*label_data_temp.shape[1]+k] = 1
					else:
						self.label_list.append(j*label_data_temp.shape[1]+k)
			
			label_data_temp = label_data_temp.reshape(label_data_temp.shape[0]*label_data_temp.shape[1], 2)
			self.label_array = np.vstack((self.label_array, label_data_temp))

		if self.one_hot_encode:
			return self.data_set, self.label_set_one_hot_encoded, self.label_array
		else:
			return self.data_set, self.label_list, self.label_array
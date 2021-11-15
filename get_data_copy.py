from scipy import io
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
import natsort
import shutil

class HandleData():                                  
	def __init__(self, phase, folder, folder_copy, one_hot_encode=False, iterations='all',until='all',samples_by_distance='all'):

		self.folder = folder
		self.folder_copy = folder_copy
		content = io.loadmat(folder + '/' + os.listdir(folder)[0])
		matrix_name = list(content.keys())[-1]
		data = content[matrix_name]
		self.phase = phase
		self.data = data[::self.phase,::self.phase,:]
		self.data_set = np.empty((0, self.data.shape[2] - 1), int)
		if self.data.shape[1] != 1:
			self.label_array = np.empty((0, 2 + 1), int)
		else:
			self.label_array = np.empty((0, 1 + 1), int)
		
		self.one_hot_encode = one_hot_encode

		if iterations == 'all':
			self.iterations = None
		else:
			self.iterations = iterations

		if until == 'all':
			self.until = None
		else:
			self.until = until

		if samples_by_distance == 'all':
			self.samples_by_distance = None
		else:
			self.samples_by_distance = samples_by_distance

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

		list_files = natsort.natsorted(os.listdir(self.folder))[0:self.until]

		list_index = [x+1 for x in range(0,self.until,self.iterations)]

		for index in list_index:
			for file in (list_files[index-1:index+self.samples_by_distance-1]):

				shutil.copyfile(self.folder + '/' + file, self.folder_copy + '/' + file)

				
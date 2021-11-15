import os
from model import HandleModel                              #to use the methods of a model (train and predict)
from get_data_copy import HandleData                            #to get the data
from save_logs import logs                                 #to save the results
from scipy import io
import numpy as np
import natsort                                             #for sorting lists naturally
from sklearn.pipeline import Pipeline
	

if __name__ == '__main__':

	Data_Folder = './DOA_Data' #Folder where the data is
	Data_Copy = './DOA_Data_Copy/'    #Folder where the results are saved

	objectLogs = logs()
	test_percentage = 0.2
	iter_distance_data = 50
	samples_distance= 50
	phase_distance = 500

	number_files = int((300-10+1)*iter_distance_data)

	phase_list = [5]
	# # print(distanceList)

	for phase in phase_list:
		print('\nphase angle: ' + str(phase))

		dirFiles = natsort.natsorted(os.listdir(Data_Folder + '/' + os.listdir(Data_Folder)[0]))

		for j in dirFiles:
			print('\nantenna number: ' + str(j))
			
			objectLogs.createFolder(Data_Copy + os.listdir(Data_Folder)[0] + '/' + str(j) + '/')

			handleData = HandleData(phase, folder = Data_Folder + '/' + os.listdir(Data_Folder)[0] + '/' + str(j), folder_copy = Data_Copy + os.listdir(Data_Folder)[0] + '/' + str(j) + '/', iterations = phase_distance, until=number_files, samples_by_distance=samples_distance) #folder = './Dround_Data_New/Nomalized'
						
			antenna_data= handleData.get_synthatic_data() 
						
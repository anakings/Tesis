import os
from model import HandleModel                              #to use the methods of a model (train and predict)
from get_data import HandleData                            #to get the data
from save_logs import logs                                 #to save the results
from sklearn.model_selection import train_test_split       #Split arrays or matrices into random train and test subsets
#from sklearn.model_selection import GridSearchCV
#from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.multioutput import MultiOutputClassifier      #This is a simple strategy for extending classifiers that do not natively support multi-target classification
from sklearn.svm import SVC
from scipy import io
import numpy as np
import natsort                                             #for sorting lists naturally



def first_proposal():
	print('Find the angle of azimuth and the angle of elevation independently' , 'Find the azimuth angle\n', sep='\n')
				
	print('Shuffling data...')
	antenna_data_train, antenna_data_test, label_data_train, label_data_test = train_test_split(antenna_data, label_matrix[:,0], test_size=test_percentage, random_state=42) 
				
	modelObject = HandleModel(antenna_data_train, antenna_data_test, label_data_train, label_data_test)

	print('Instantiating the model...')
	#clf = GridSearchCV(model, param_grid, cv=5, verbose=3, n_jobs=-1)
	clf = model
		
	accuracyOfModel_azimuth, time_validation_azimuth = modelObject.train_predict(clf, modelName[name])
	print('Accuracy of azimuth model', modelName[name], 'is:', accuracyOfModel_azimuth)

	print('\nFind the elevation angle\n')
	print('Shuffling data...')
	antenna_data_train, antenna_data_test, label_data_train, label_data_test = train_test_split(antenna_data, label_matrix[:,1], test_size=test_percentage, random_state=42) 
				
	modelObject = HandleModel(antenna_data_train, antenna_data_test, label_data_train, label_data_test)
		
	accuracyOfModel_elevation, time_validation_elevation = modelObject.train_predict(clf, modelName[name])
	print('Accuracy of elevation model', modelName[name], 'is:', accuracyOfModel_elevation)

	#add to file Logs_Folder + str(i) + '/' + model: j and the accuracy
	print('\nAccuracy of 1st proposal with', modelName[name], 'is', accuracyOfModel_azimuth*accuracyOfModel_elevation/100)
	listValues = [j, accuracyOfModel_azimuth*accuracyOfModel_elevation/100, time_validation_azimuth + time_validation_elevation]
	objectLogs.writeFile(Logs_Folder + 'first_proposal/' + str(i) + '/' + modelName[name] + '.csv', listValues)


def second_proposal():
	print('\nFind the azimuth angle and the elevation angle with the same ML model instance.\n')
	print('Shuffling data...')
	antenna_data_train, antenna_data_test, label_data_train, label_data_test = train_test_split(antenna_data, label_list, test_size=test_percentage, random_state=42) 
				
	modelObject = HandleModel(antenna_data_train, antenna_data_test, label_data_train, label_data_test)

	print('Instantiating the model...')
	#clf = GridSearchCV(model, param_grid, cv=5, verbose=3, n_jobs=-1)
	clf = model
		
	accuracyOfModel, time_validation = modelObject.train_predict(clf, modelName[name])
	print('Accuracy of 2nd proposal with', modelName[name], 'is', accuracyOfModel)
				
	#add to file Logs_Folder + str(i) + '/' + model: j and the accuracy
	listValues = [j, accuracyOfModel, time_validation]
	objectLogs.writeFile(Logs_Folder+ 'second_proposal/' + str(i) + '/' + modelName[name] + '.csv', listValues)

def third_proposal():
	print('\nFind the azimuth angle and the elevation angle with the output ML model.\n')
	print('Shuffling data...')
	antenna_data_train, antenna_data_test, label_data_train, label_data_test = train_test_split(antenna_data, label_matrix, test_size=test_percentage, random_state=42) 

	modelObject = HandleModel(antenna_data_train, antenna_data_test, label_data_train, label_data_test)

	print('Instantiating the model...')
	#clf = GridSearchCV(model, param_grid_list, cv=5, verbose=3, n_jobs=-1)
	clf = MultiOutputClassifier(model)

	accuracyOfModel, time_validation = modelObject.train_predict(clf, modelName[name])
	print('Accuracy of 3rd proposal with', modelName[name], 'is', accuracyOfModel)
				
	#add to file Logs_Folder + str(i) + '/' + model: j and the accuracy
	listValues = [j, accuracyOfModel, time_validation]
	objectLogs.writeFile(Logs_Folder+ 'third_proposal/' + str(i) + '/' + modelName[name] + '.csv', listValues)


if __name__ == '__main__':

	Data_Folder = './DOA_Data' #Folder where the data is
	Logs_Folder = './Logs/'    #Folder where the results are saved

	objectLogs = logs()
	test_percentage = 0.2
	phase = 5

	
	modelList = [DecisionTreeClassifier(random_state=42)]
	modelName = ['DT']
	
	'''
	modelList = [DecisionTreeClassifier(max_depth=900, max_features='auto', random_state=42)]
	modelName = ['DT']

	modelList = [RandomForestClassifier(n_estimators=10)]
	modelName = ['RF']
	'''
	
	for name,model in enumerate(modelList):
		for i in os.listdir(Data_Folder):
			print('\n' + i)
			dirFiles = os.listdir(Data_Folder + '/' +  str(i))
			sortedDirFiles = natsort.natsorted(dirFiles)
			
			
			#create a folder Logs_Folder + str(i)
			objectLogs.createFolder(Logs_Folder + 'first_proposal/' + str(i))
			objectLogs.createFolder(Logs_Folder + 'second_proposal/' + str(i))
			objectLogs.createFolder(Logs_Folder + 'third_proposal/' + str(i))
			

			#clean the file: modelName[name] + '.csv'
			#open(Logs_Folder + 'first_proposal/' + str(i) + '/' + modelName[name] + '.csv', 'w').close()
			#open(Logs_Folder + 'second_proposal/' + str(i) + '/' + modelName[name] + '.csv', 'w').close()
			#open(Logs_Folder + 'third_proposal/' + str(i) + '/' + modelName[name] + '.csv', 'w').close()
			

			for j in sortedDirFiles:
				print('\n' + j)
				
				handleData = HandleData(phase, folder = Data_Folder + '/' + str(i) + '/' + str(j), iterations = 'all') #folder = './Dround_Data_New/Nomalized'
				
				print('Getting data...')
				antenna_data, label_list, label_matrix = handleData.get_synthatic_data() #label_list: for the second proposal, label_matrix: for the first and third proposal 
				
				first_proposal_response = first_proposal()
				
				#if (modelName[name] != 'Bagging Classifier'):
					#second_proposal_response = second_proposal()
				
				#third_proposal_response = third_proposal()
						
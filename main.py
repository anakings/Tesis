import os
from model import HandleModel                              #to use the methods of a model (train and predict)
from get_data import HandleData                            #to get the data
from save_logs import logs                                 #to save the results
from sklearn.model_selection import train_test_split       #Split arrays or matrices into random train and test subsets
#from sklearn.model_selection import GridSearchCV
#from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.multioutput import MultiOutputClassifier      #This is a simple strategy for extending classifiers that do not natively support multi-target classification
from sklearn.svm import SVC
from scipy import io
import numpy as np
import natsort                                             #for sorting lists naturally
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,  RidgeClassifier, SGDClassifier
from sklearn.pipeline import Pipeline



def first_proposal():
	print('Find the angle of azimuth and the angle of elevation independently' , 'Find the azimuth angle\n', sep='\n')
				
	print('Shuffling data...')
	# labels = label_matrix[:,0].reshape(label_matrix[:,0].shape[0], 1)
	# enc = OneHotEncoder(handle_unknown='ignore')
	# enc.fit(labels)
	# labels = enc.transform(labels).toarray()
	# antenna_data_train, antenna_data_test, label_data_train, label_data_test = train_test_split(antenna_data, labels, test_size=test_percentage, random_state=42)
	antenna_data_train, antenna_data_test, label_data_train, label_data_test = train_test_split(antenna_data, label_matrix[:,0], test_size=test_percentage, random_state=42)

	# clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
	# clf.fit(antenna_data_train, label_data_train)
	# clf.score(antenna_data_test, label_data_test)*100
				
	modelObject = HandleModel(antenna_data_train, antenna_data_test, label_data_train, label_data_test)
	print(label_data_test)

	print('Instantiating the model...')
	#clf = GridSearchCV(model, param_grid, cv=5, verbose=3, n_jobs=-1)
	clf = model
		
	accuracyOfModel_azimuth, time_validation_azimuth = modelObject.train_predict(clf, modelName[name])
	print('Accuracy of azimuth model', modelName[name], 'is:', accuracyOfModel_azimuth)

	print('\nFind the elevation angle\n')
	print('Shuffling data...')
	# labels = label_matrix[:,1].reshape(label_matrix[:,1].shape[0], 1)
	# enc = OneHotEncoder(handle_unknown='ignore')
	# enc.fit(labels)
	# labels = enc.transform(labels).toarray()
	antenna_data_train, antenna_data_test, label_data_train, label_data_test = train_test_split(antenna_data, label_matrix[:,1], test_size=test_percentage, random_state=42) 
				
	modelObject = HandleModel(antenna_data_train, antenna_data_test, label_data_train, label_data_test)
		
	accuracyOfModel_elevation, time_validation_elevation = modelObject.train_predict(clf, modelName[name])
	print('Accuracy of elevation model', modelName[name], 'is:', accuracyOfModel_elevation)

	'''

	print('\nFind the distance\n')
	print('Shuffling data...')
	# labels = label_matrix[:,2].reshape(label_matrix[:,2].shape[0], 1)
	# enc = OneHotEncoder(handle_unknown='ignore')
	# enc.fit(labels)
	# labels = enc.transform(labels).toarray()
	antenna_data_train, antenna_data_test, label_data_train, label_data_test = train_test_split(antenna_data, label_matrix[:,2], test_size=test_percentage, random_state=42) 
				
	modelObject = HandleModel(antenna_data_train, antenna_data_test, label_data_train, label_data_test)
		
	accuracyOfModel_distance, time_validation_distance = modelObject.train_predict(clf, modelName[name])
	print('Accuracy of distance model', modelName[name], 'is:', accuracyOfModel_distance)

	#add to file Logs_Folder + str(i) + '/' + model: j and the accuracy
	# print('\nAccuracy of 1st proposal with', modelName[name], 'is', accuracyOfModel_azimuth*accuracyOfModel_elevation/100)
	# listValues = [j, accuracyOfModel_azimuth*accuracyOfModel_elevation/100, time_validation_azimuth + time_validation_elevation]
	# objectLogs.writeFile(Logs_Folder + str(i) + '/' + str(j) + '/first_proposal/' + modelName[name] + '.csv', listValues)

	print('\nAccuracy of 1st proposal with', modelName[name], 'is', accuracyOfModel_azimuth*accuracyOfModel_elevation*accuracyOfModel_distance/10000)
	listValues = [j, accuracyOfModel_azimuth*accuracyOfModel_elevation*accuracyOfModel_distance/10000, time_validation_azimuth + time_validation_elevation,accuracyOfModel_azimuth,accuracyOfModel_elevation,accuracyOfModel_distance]
	return(listValues)
	'''
	print('\nAccuracy of 1st proposal with', modelName[name], 'is', accuracyOfModel_azimuth*accuracyOfModel_elevation/100)
	listValues = [j, accuracyOfModel_azimuth*accuracyOfModel_elevation/100, time_validation_azimuth + time_validation_elevation,accuracyOfModel_azimuth,accuracyOfModel_elevation]
	return(listValues)



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

	objectLogs.writeFile(Logs_Folder + str(i) + '/' + 'second_proposal/' + modelName[name] + '.csv', listValues)

def third_proposal():
	print('\nFind the azimuth angle and the elevation angle with the output ML model.\n')
	print('Shuffling data...')
	antenna_data_train, antenna_data_test, label_data_train, label_data_test = train_test_split(antenna_data, label_matrix[:,:-1], test_size=test_percentage, random_state=42) 

	modelObject = HandleModel(antenna_data_train, antenna_data_test, label_data_train, label_data_test)

	print('Instantiating the model...')
	#clf = GridSearchCV(model, param_grid_list, cv=5, verbose=3, n_jobs=-1)
	#clf = MultiOutputClassifier(model)
	clf = model

	accuracyOfModel, time_validation = modelObject.train_predict(clf, modelName[name])
	print('Accuracy of 3rd proposal with', modelName[name], 'is', accuracyOfModel)
				
	#add to file Logs_Folder + str(i) + '/' + model: j and the accuracy
	listValues = [j, accuracyOfModel, time_validation]
	return(listValues)
	

if __name__ == '__main__':

	Data_Folder = './DOA_Data' #Folder where the data is
	Logs_Folder = './Logs/'    #Folder where the results are saved

	objectLogs = logs()
	test_percentage = 0.2
	iter_distance_data = 50
	samples_distance= 50
	phase_files = 500

	#modelList = [SGDRegressor(max_iter=1000,alpha=0.9)]
	#modelName = ['SGDRegressor']
	'''
	modelList = [DecisionTreeClassifier(random_state=42)]
	modelName = ['DT']
	
	modelList = [DecisionTreeRegressor(random_state=42)]
	modelName = ['DT_Regressor']
	'''
	# modelList = [DecisionTreeClassifier(max_depth=900, max_features='auto', random_state=42)]
	# modelName = ['DT']

	# modelList = [RandomForestClassifier(n_estimators=10)]
	# modelName = ['RF']
	modelList = [LinearRegression()]
	modelName = ['LinearRegression']

	min_distance = 10
	max_distance_range = 300
	min_distance_range = 100
	distanceList = np.linspace(min_distance_range, max_distance_range, num = 3, dtype = int)

	phase_list = np.linspace(5, 20, num = 4, dtype = int)
	# # print(distanceList)

	for phase in phase_list:
		print('\nphase angle: ' + str(phase))
		for distanceValue in distanceList:
			print('\nmax distance: ' + str(distanceValue))
			number_files = int((distanceValue-min_distance+1)*iter_distance_data)
			print('total data: ' + str(number_files))

			dirFiles = natsort.natsorted(os.listdir(Data_Folder + '/' + os.listdir(Data_Folder)[0]))
			
			for name,model in enumerate(modelList):

				for j in dirFiles:
					print('\nantenna number: ' + str(j))
					
					#create a folder Logs_Folder + str(i)
					objectLogs.createFolder(Logs_Folder + str(phase) + '/'  + str(distanceValue) + '/' + str(j) + '/first_proposal/')
					#objectLogs.createFolder(Logs_Folder + 'second_proposal/' + str(i))
					objectLogs.createFolder(Logs_Folder + str(phase) + '/' + str(distanceValue) + '/' + str(j) + '/third_proposal/')
					
					'''
					#clean the file: modelName[name] + '.csv'
					open(Logs_Folder + str(phase) + '/'  + str(distanceValue) + '/' + str(j) + '/first_proposal/' + modelName[name] + '.csv', 'w').close()
					#open(Logs_Folder + 'second_proposal/' + str(i) + '/' + modelName[name] + '.csv', 'w').close()
					open(Logs_Folder + str(phase) + '/' + str(distanceValue) + '/' + str(j) + '/third_proposal/' + modelName[name] + '.csv', 'w').close()
					'''
					handleData = HandleData(phase, folder = Data_Folder + '/' + os.listdir(Data_Folder)[0] + '/' + str(j), phase_distance = phase_files, until=number_files, samples_by_distance=samples_distance) #folder = './Dround_Data_New/Nomalized'
						
					print('Getting data...')
					antenna_data, label_list, label_matrix = handleData.get_synthatic_data() #label_list: for the second proposal, label_matrix: for the first and third proposal 	
				
					first_proposal_response = first_proposal()
					objectLogs.writeFile(Logs_Folder + str(phase) + '/' + str(distanceValue) + '/' + str(j) + '/first_proposal/' + modelName[name] + '.csv', first_proposal_response)
						
					#if (modelName[name] != 'Bagging Classifier'):
						#second_proposal_response = second_proposal()
						
					third_proposal_response = third_proposal()
					objectLogs.writeFile(Logs_Folder + str(phase) + '/' + str(distanceValue) + '/' + str(j) + '/third_proposal/' + modelName[name] + '.csv', third_proposal_response)
				
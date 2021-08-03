from sklearn.model_selection import train_test_split
import os
from model import HandleModel
from get_data import HandleData
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
#from sklearn.tree import ExtraTreeClassifier 
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from scipy import io
import numpy as np
import natsort
from save_logs import logs
from sklearn.multioutput import MultiOutputClassifier

def first_proposal():
	print('Find the angle of azimuth and the angle of elevation independently' , 'Find the azimuth angle\n', sep='\n')
				
	print('Shuffling data...')
	antenna_data_train, antenna_data_test, label_data_train, label_data_test = train_test_split(antenna_data, label_matrix[:,0], test_size=test_percentage, random_state=42) 
				
	modelObject = HandleModel(antenna_data_train, antenna_data_test, label_data_train, label_data_test)

	print('Instantiating the model...')
	clf = GridSearchCV(model, param_grid_list[name], cv=5, verbose=3, n_jobs=-1)
	#clf = model
		
	accuracyOfModel_azimuth, time_validation_azimuth, best_param_azimuth = modelObject.train_predict_GridSearch(clf, modelName[name])
	print('Accuracy of azimuth model', modelName[name], 'is:', accuracyOfModel_azimuth)

	print('\nFind the elevation angle\n')
	print('Shuffling data...')
	antenna_data_train, antenna_data_test, label_data_train, label_data_test = train_test_split(antenna_data, label_matrix[:,1], test_size=test_percentage, random_state=42) 
				
	modelObject = HandleModel(antenna_data_train, antenna_data_test, label_data_train, label_data_test)
		
	accuracyOfModel_elevation, time_validation_elevation, best_param_elevation = modelObject.train_predict_GridSearch(clf, modelName[name])
	print('Accuracy of elevation model', modelName[name], 'is:', accuracyOfModel_elevation)

	#add to file Logs_Folder + str(i) + '/' + model: j and the accuracy
	print('\nAccuracy of 1st proposal with', modelName[name], 'is', accuracyOfModel_azimuth*accuracyOfModel_elevation/100)
	listValues = [j, accuracyOfModel_azimuth*accuracyOfModel_elevation/100, time_validation_azimuth + time_validation_elevation, best_param_azimuth, best_param_elevation]
	objectLogs.writeFile(Logs_Folder + 'first_proposal/' + str(i) + '/' + modelName[name] + '.csv', listValues)


def second_proposal():
	print('\nFind the azimuth angle and the elevation angle with the same ML model instance.\n')
	print('Shuffling data...')
	antenna_data_train, antenna_data_test, label_data_train, label_data_test = train_test_split(antenna_data, label_list, test_size=test_percentage, random_state=42) 
				
	modelObject = HandleModel(antenna_data_train, antenna_data_test, label_data_train, label_data_test)

	print('Instantiating the model...')
	clf = GridSearchCV(model, param_grid_list[name], cv=5, verbose=3, n_jobs=-1)
	#clf = model
		
	accuracyOfModel, time_validation, best_param = modelObject.train_predict_GridSearch(clf, modelName[name])
	print('Accuracy of 2nd proposal with', modelName[name], 'is', accuracyOfModel)
				
	#add to file Logs_Folder + str(i) + '/' + model: j and the accuracy
	listValues = [j, accuracyOfModel, time_validation, best_param]
	objectLogs.writeFile(Logs_Folder+ 'second_proposal/' + str(i) + '/' + modelName[name] + '.csv', listValues)

def third_proposal():
	print('\nFind the azimuth angle and the elevation angle with the output ML model.\n')
	print('Shuffling data...')
	antenna_data_train, antenna_data_test, label_data_train, label_data_test = train_test_split(antenna_data, label_matrix, test_size=test_percentage, random_state=42)

	modelObject = HandleModel(antenna_data_train, antenna_data_test, label_data_train, label_data_test)

	print('Instantiating the model...')
	
	clf = GridSearchCV(MultiOutputClassifier(model), param_grid_list[name], cv=5, verbose=3, n_jobs=-1)

	accuracyOfModel, time_validation, best_param = modelObject.train_predict_GridSearch(clf, modelName[name])
	print('Accuracy of 3rd proposal with', modelName[name], 'is', accuracyOfModel)
				
	#add to file Logs_Folder + str(i) + '/' + model: j and the accuracy
	listValues = [j, accuracyOfModel, time_validation, best_param]
	objectLogs.writeFile(Logs_Folder+ 'third_proposal/' + str(i) + '/' + modelName[name] + '.csv', listValues)


if __name__ == '__main__':

	Data_Folder = './DOA_Data'
	Logs_Folder = './GridSearch/Logs/'

	objectLogs = logs()
	test_percentage = 0.2
	phase = 5

	'''
	modelList = [DecisionTreeClassifier(random_state=42), BaggingClassifier(DecisionTreeClassifier(random_state=42), bootstrap=False, n_jobs=-1, random_state=42), SVC()]
	modelName = ['Decision Tree Classifier', 'Bagging Classifier', 'SVC']
	
	param_grid_list = [[{'max_depth': [x for x in range(1,25)], 'min_samples_split': np.linspace(0.1, 1.0, 10)},
	{'max_depth': [x for x in range(1,25)]}],[{'n_estimators': [x for x in range(100,1100,100)], 'max_samples': np.linspace(0.1, 1.0, 5)}],[{'kernel': ['rbf','poly','sigmoid'], 'gamma': np.logspace(-6, 10, 5), 'C': [1, 10, 100, 1000,10000]},
						{'kernel': ['linear'], 'C': [1, 10, 100, 1000,10000]}]]
	
	
	

	modelList = [RandomForestClassifier(DecisionTreeClassifier())]
	modelName = ['RF']
	#param_grid_list = [{'estimator__max_depth': [x for x in range(10,110,10)]}]
	param_grid_list = [{'n_estimators': [x for x in range(10,60,10)]}]
	'''

	modelList = [DecisionTreeClassifier(random_state=42)]
	modelName = ['DT']
	#param_grid_list = [{'max_depth': [x for x in range(100,1100,100)]}]
	param_grid_list = [{'estimator__max_depth': [x for x in range(100,1100,100)]}]

	
	for name,model in enumerate(modelList):
		for i in os.listdir(Data_Folder):
			print('\n' + i)
			dirFiles = os.listdir(Data_Folder + '/' +  str(i))
			sortedDirFiles = natsort.natsorted(dirFiles)

			
			#create a folder Logs_Folder + str(i)
			#objectLogs.createFolder(Logs_Folder + 'first_proposal/' + str(i))
			#objectLogs.createFolder(Logs_Folder + 'second_proposal/' + str(i))
			#objectLogs.createFolder(Logs_Folder + 'third_proposal/' + str(i))

			#clean the file: modelName[name] + '.csv'
			#open(Logs_Folder + 'first_proposal/' + str(i) + '/' + modelName[name] + '.csv', 'w').close()
			#(Logs_Folder + 'second_proposal/' + str(i) + '/' + modelName[name] + '.csv', 'w').close()
			open(Logs_Folder + 'third_proposal/' + str(i) + '/' + modelName[name] + '.csv', 'w').close()
			

			for j in sortedDirFiles:
			#for j in [5,6,7,8,9,10,11,12,13,14,15,16]:
				print('\n' + str(j))
				
				handleData = HandleData(phase, folder = Data_Folder + '/' + str(i) + '/' + str(j), iterations = 'all') #folder = './Dround_Data_New/Nomalized'

				print('Getting data...')
				antenna_data, label_list, label_matrix = handleData.get_synthatic_data()
				
				#first_proposal_response = first_proposal()
				
				#if (modelName[name] != 'Bagging Classifier'):
					#second_proposal_response = second_proposal()
				
				third_proposal_response = third_proposal()
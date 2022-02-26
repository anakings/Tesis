import os
#from model import HandleModel                              #to use the methods of a model (train and predict)
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
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, multilabel_confusion_matrix
import seaborn as sns

def plot_confusion_matrix(true_label, predicted_label, step, limit, folder_save):
	# Plot the confusion matrix.
	mat = confusion_matrix(true_label, predicted_label)

	classes = range(mat.shape[0])

	plt.figure(figsize=(25,15))
	sns.heatmap(mat.T[0::step,0::step], square=True, annot=True, fmt='d', cbar=False, xticklabels=classes[0::step], yticklabels=classes[0::step], cmap="Blues")
	plt.xlabel('true label')
	plt.ylabel('predicted label')
	plt.savefig(folder_save + 'confusion_matrix ' + modelName[name] + '.png')
	plt.show()

	# Plot the confusion matrix (limit).
	plt.figure(figsize=(15,9))
	sns.heatmap(mat.T[0:limit,0:limit], square=True, annot=True, fmt='d', cbar=False, xticklabels=range(limit), yticklabels=range(limit), cmap="Blues")
	plt.xlabel('true label')
	plt.ylabel('predicted label')
	plt.savefig(folder_save + 'confusion_matrix_limit ' + modelName[name] + '.png')
	plt.show()

def plot_histogram(y_pred, y, x, folder_save, string_angle, x_lim, x_label, y_label):
	#print(y)
	y = y.ravel()

	print(y_pred.shape)
	print(y.shape)
	x_axis = []
	x_axis1 = []
	#dif_list = []
	for i in range(len(y)):
		if(y[i] != y_pred[i]):
			if(np.abs(float(y[i]) - float(y_pred[i])) > 10.0):
				print('index:', i, ' - target:', y[i], ' - predicted:', y_pred[i], ' - diff:', np.abs(float(y[i]) - float(y_pred[i])), ' -', string_angle, ':',  x[i])
				x_axis.append(x[i])
				#dif_list.append(np.abs(float(y[i]) - float(y_pred[i])))
	#print(sum(elevation_list)/len(elevation_list))
	#print(sorted(x_list))
	#print(sum(dif_list)/len(elevation_list))
	
	error_histogram = [0 for i in range(x_lim)]
	for i in x_axis:
		error_histogram[int(i)] = error_histogram[int(i)] + 1

	plt.plot([i for i in range(x_lim)], error_histogram, '.')
	plt.ylim(0, 600)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.savefig(folder_save + string_angle + '_histogram ' + modelName[name] + '.png')
	plt.show()
	

def independently_proposal(folder_save):
	print('Find the angle of azimuth and the angle of elevation independently')
	
	#print('\nFind the azimuth angle\n')			
	antenna_data_train, antenna_data_test, label_train, label_test = train_test_split(antenna_data, label_matrix[:,0], test_size=test_percentage, random_state=42)
	clf = model
	clf.fit(antenna_data_train,label_train)
	label_predicted = clf.predict(antenna_data_test)
	accuracyOfModel_azimuth = clf.score(antenna_data_test,label_test)	
	print('Accuracy of azimuth model', modelName[name], 'is:', accuracyOfModel_azimuth)
	plot_matrix = plot_confusion_matrix(label_test, label_predicted, 5, 20, folder_save + 'azimuth_')
	#histogram = plot_histogram(clf.predict(antenna_data), label_matrix[:,0], label_matrix[:,1], folder_save, 'elevation', 91)

	#print('\nFind the elevation angle\n')
	antenna_data_train, antenna_data_test, label_train, label_test = train_test_split(antenna_data, label_matrix[:,1], test_size=test_percentage, random_state=42) 
	clf.fit(antenna_data_train,label_train)
	label_predicted = clf.predict(antenna_data_test)
	accuracyOfModel_elevation = clf.score(antenna_data_test,label_test)			
	print('Accuracy of elevation model', modelName[name], 'is:', accuracyOfModel_elevation)
	plot_matrix = plot_confusion_matrix(label_test, label_predicted, 5, 20, folder_save + 'elevation_')
	#histogram = plot_histogram(clf.predict(antenna_data), label_matrix[:,1], label_matrix[:,0], folder_save, 'azimuth', 361)
	
	print('\nAccuracy of 1st proposal with', modelName[name], 'is', accuracyOfModel_azimuth*accuracyOfModel_elevation)
	#listValues = [j, accuracyOfModel_azimuth*accuracyOfModel_elevation/100, time_validation_azimuth + time_validation_elevation,accuracyOfModel_azimuth,accuracyOfModel_elevation]
	listValues = [j, accuracyOfModel_azimuth*accuracyOfModel_elevation,accuracyOfModel_azimuth,accuracyOfModel_elevation]
	objectLogs.writeFile(folder_save + modelName[name] + '.csv', listValues)


def multioutput_proposal(folder_save):
	print('\nFind the azimuth angle and the elevation angle with the output ML model.\n')
	antenna_data_train, antenna_data_test, label_train, label_test = train_test_split(antenna_data, label_matrix, test_size=test_percentage, random_state=42) 

	#clf = GridSearchCV(model, param_grid_list, cv=5, verbose=3, n_jobs=-1)
	clf = MultiOutputClassifier(model)
	clf.fit(antenna_data_train,label_train)
	label_predicted = clf.predict(antenna_data_test)
	
	accuracyOfModel = clf.score(antenna_data_test,label_test)
	print('Accuracy of 3rd proposal with', modelName[name], 'is', accuracyOfModel)
	
	plot_matrix = plot_confusion_matrix(label_test[:,0], label_predicted[:,0], 5, 20, folder_save + 'azimuth_')
	plot_matrix = plot_confusion_matrix(label_test[:,1], label_predicted[:,1], 5, 20, folder_save + 'elevation_')
	histogram = plot_histogram(clf.predict(antenna_data)[:,0], label_matrix[:,0], label_matrix[:,1], folder_save, 'elevation', 91, 'elevation angle', 'Number of incorrectly predicted azimuth angles')
	histogram = plot_histogram(clf.predict(antenna_data)[:,0], label_matrix[:,0], label_matrix[:,0], folder_save, 'azimuth', 361, 'azimuth angle', 'Number of incorrectly predicted azimuth angles')
	#histogram = plot_histogram(clf.predict(antenna_data)[:,1], label_matrix[:,1], label_matrix[:,0], folder_save, 'azimuth', 361, 'azimuth angle', 'Number of incorrectly predicted elevation angles')
	histogram = plot_histogram(clf.predict(antenna_data)[:,1], label_matrix[:,1], label_matrix[:,0], folder_save, 'elevation', 91, 'elevation angle', 'Number of incorrectly predicted elevation angles')
			
	#add to file Logs_Folder + str(i) + '/' + model: j and the accuracy
	listValues = [j, accuracyOfModel]
	objectLogs.writeFile(folder_save + modelName[name] + '.csv', listValues)

if __name__ == '__main__':
	
	Data_Folder = './dipole_antenna_plus' #Folder where the data is
	Logs_Folder = './results_dipoleVee/'    #Folder where the results are saved
	'''
	
	Data_Folder = './dipole_antenna_plus' #Folder where the data is
	Logs_Folder = './results_dipole_antenna_plus/'    #Folder where the results are saved
	'''
	objectLogs = logs()
	test_percentage = 0.2
	iter_distance_data = 10                     #How many files are there for the same distance
	samples_distance= 10                         #sHow many files of the same distance do you want to take. How many files are to be taken from the file pointed to by the phase_files pointer.
	phase_files = 100                            #Point from which file you want to take the data, giving jumps equal to the specified number.
	
	modelList = [DecisionTreeClassifier(random_state=42)]
	modelName = ['DT_Classifier']

	#modelList = [SGDRegressor(max_iter=1000,alpha=0.9)]
	#modelName = ['SGDRegressor']
	
	#modelList = [DecisionTreeClassifier(max_depth=300, max_features='auto', random_state=42)]
	#modelName = ['DT']
	
	# modelList = [DecisionTreeClassifier(max_depth=900, max_features='auto', random_state=42)]
	# modelName = ['DT']

	#modelList = [RandomForestClassifier(n_estimators=10)]
	#modelName = ['RF']
	#modelList = [LinearRegression()]
	#modelName = ['LinearRegression']

	min_distance = 123
	distanceList = [213]

	phase_list = [1]
	#list_proposal = [independently_proposal, multioutput_proposal]
	list_proposal = [multioutput_proposal]

	for phase in phase_list:
		print('\nphase angle: ' + str(phase))
		for distanceValue in distanceList:
			print('\nmax distance: ' + str(distanceValue))
			number_files = int((distanceValue-min_distance+1)*iter_distance_data)        #Up to which file you want to select (up to what distance you want to select)
			#print('total data: ' + str(number_files))
			#number_files = 500

			dirFiles = natsort.natsorted(os.listdir(Data_Folder + '/' + os.listdir(Data_Folder)[0]))
			
			for name,model in enumerate(modelList):

				for j in dirFiles:
				#for j in [4]:
					print('\nantenna number: ' + str(j))

					for proposal in list_proposal:
						
						#print(proposal.__name__)
						folder_save = Logs_Folder + str(phase) + '/'  + str(distanceValue) + '/' + str(j) + '/' + proposal.__name__ + '/'
					
						#create a folder Logs_Folder + str(i)
						objectLogs.createFolder(folder_save)
						
						#clean the file:folder_save + modelName[name] + '.csv'
						open(folder_save + modelName[name] + '.csv', 'w').close()
						
						handleData = HandleData(phase, folder = Data_Folder + '/' + os.listdir(Data_Folder)[0] + '/' + str(j), phase_distance = phase_files, until=number_files, samples_by_distance=samples_distance) #folder = './Dround_Data_New/Nomalized'
						#handleData = HandleData(phase, folder = Data_Folder + '/' + os.listdir(Data_Folder)[0] + '/' + str(j), phase_distance = phase_files, until=800, samples_by_distance=samples_distance) #folder = './Dround_Data_New/Nomalized'
							
						print('\nGetting data...')
						antenna_data, label_list, label_matrix = handleData.get_synthatic_data() #label_list: for the second proposal, label_matrix: for the first and third proposal 	
						#for a in label_matrix:
						#	print(a)
						print(label_matrix)
						print(label_matrix.shape)
						'''
						for ttt in label_matrix:
							print(ttt)
						'''
						label_matrix = label_matrix[:,:-2]               #Cojo solo el angulo de azimuth y de elevacion, o sea no uso ni el cuadrante ni la distancia
						
						
						antenna_data = antenna_data[:,:-1]               #Quito la antena del centro del sistema. Si la quiero dejar tengo q comentarear esat parte
						v = np.sum(antenna_data, axis = 1)               #Suma de las potencias de las antenas en cada ciclo
						v = np.reshape(v, (v.shape[0], 1))               #
						antenna_data = antenna_data/v                    #normalizo las potencias
						
						
						ttt = proposal(folder_save)
						
						
						'''
						#print(proposal.__name__)
						folder_save = Logs_Folder1 + str(phase) + '/'  + str(distanceValue) + '/' + str(j) + '/' + proposal.__name__ + '/'
					
						#create a folder Logs_Folder + str(i)
						objectLogs.createFolder(folder_save)
						
						#clean the file:folder_save + modelName[name] + '.csv'
						open(folder_save + modelName[name] + '.csv', 'w').close()
						
						handleData = HandleData(phase, folder = Data_Folder + '/' + os.listdir(Data_Folder)[0] + '/' + str(j), phase_distance = phase_files, until=number_files, samples_by_distance=samples_distance) #folder = './Dround_Data_New/Nomalized'
							
						print('\nGetting data...')
						antenna_data, label_list, label_matrix = handleData.get_synthatic_data() #label_list: for the second proposal, label_matrix: for the first and third proposal 	
						label_matrix = label_matrix[:,:-2]
						ttt = proposal(folder_save)
						'''
						
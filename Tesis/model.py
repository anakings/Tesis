from time import time
import numpy as np

class HandleModel():
	
	def __init__(self, antenna_data_train, antenna_data_test, label_data_train, label_data_test):

		self.antenna_data_train = antenna_data_train 
		self.antenna_data_test = antenna_data_test
		self.label_data_train = label_data_train 
		self.label_data_test = label_data_test
 
	def train_predict(self, clf, clfName):                                                             #For the main
		tic = time()
		print('Training', clfName + '...')
		clf.fit(self.antenna_data_train, self.label_data_train)   
		time_training = time() - tic
		print('Elapsed time training:', time_training, 'seconds')
		#import graphviz 
		#dot_data = tree.export_graphviz(clf, out_file=None) 
		#graph = graphviz.Source(dot_data) 
		#graph.render("DecisionTree") 
		tic = time()
		print('Predicting...')
		y_pred = clf.predict(self.antenna_data_test)
		time_validation = time() - tic
		print('Elapsed time validation:', time_validation, 'seconds')
		'''
		z=0
		t=0
		for i in range(0,y_pred.shape[0]):
			if y_pred[i,0]==self.label_data_test[i,0] and y_pred[i,1]==self.label_data_test[i,1]:
				z += 1
		z /= y_pred.shape[0]
		
		print(z)
		
		
		print(np.mean(np.all(self.label_data_test == y_pred, axis=1)))
		'''
		return clf.score(self.antenna_data_test, self.label_data_test)*100, time_validation

	def train_predict_GridSearch(self, clf, clfName):                                                    #GridSearch 
		
		tic = time()
		print('Training', clfName + '...')
		clf.fit(self.antenna_data_train, self.label_data_train)
		time_training = time() - tic
		print('Elapsed time training:', time_training, 'seconds')
		#import graphviz 
		#dot_data = tree.export_graphviz(clf, out_file=None) 
		#graph = graphviz.Source(dot_data) 
		#graph.render("DecisionTree") 
		tic = time()
		print('Predicting...')
		y_pred = clf.predict(self.antenna_data_test)
		time_validation = time() - tic
		print('Elapsed time validation:', time_validation, 'seconds')

		print(clf.best_estimator_)

		return clf.score(self.antenna_data_test, self.label_data_test)*100, time_validation, clf.best_estimator_

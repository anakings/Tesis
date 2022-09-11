from sklearn.metrics import mean_squared_error, accuracy_score

# This is a simple strategy for extending classifiers that do not natively support multi-target classification
from sklearn.multioutput import MultiOutputClassifier

import numpy as np
import pandas as pd
import copy
from sklearn.tree import DecisionTreeClassifier

class HandleModel():                                  
	def __init__(self, clf, data_train, label_train):
		self.clf = clf
		self.label_train = label_train
		self.data_train = data_train
	
	def train_model(self):
		self.clf.fit(self.data_train, self.label_train)
		
	def predict_model(self, data_test):
		label_predicted = self.clf.predict(data_test)
		
		# Convert self.label_predict to a dataframe with the same columns as self.label_train
		self.label_predicted = pd.DataFrame(label_predicted, columns=self.label_train.columns)	

class Multioutput(HandleModel):
	def __init__(self, clf, data_train, label_train):
		super().__init__(clf, data_train, label_train)
		#self.clf = MultiOutputClassifier(clf)
		self.clf = clf
		
	def score_proposal(self, data_test, label_test):
		###############################################################################################################
		############################### Find the results for target ###################################################
		accuracy_per_target = np.zeros(len(label_test.columns))
		mean_squared_error_per_target = np.zeros(len(label_test.columns))
		for index, target in enumerate(label_test.columns):
			accuracy_per_target[index] = accuracy_score(label_test[target], self.label_predicted[target])
			mean_squared_error_per_target[index] = mean_squared_error(label_test[target], self.label_predicted[target])
		###############################################################################################################

		###############################################################################################################
		############################### Find ML model results #########################################################
		accuracyOfModel = self.clf.score(data_test, label_test)
		print('Accuracy of multioutput proposal is', accuracyOfModel)
		
		MSEmodel = np.sum(mean_squared_error_per_target)
		print('Mean Squared Error of multioutput proposal is', MSEmodel)
		###############################################################################################################
		
		return accuracyOfModel, MSEmodel, accuracy_per_target, mean_squared_error_per_target

class Independently(HandleModel):
	"""docstring for multioutput"""
	def __init__(self, clf, data_train, label_train):
		super().__init__(clf, data_train, label_train)

		self.ML_models_list = []
		for index in range(len(self.label_train.columns)):
			self.ML_models_list.append(copy.deepcopy(clf))
	
	def train_model(self):
		for index, ML_model in enumerate(self.ML_models_list):
			ML_model.fit(self.data_train, self.label_train.iloc[:,index]) 	

	def predict_model(self, data_test):
		self.target_predict = []
		for target, ML_model in enumerate(self.ML_models_list):
			self.target_predict.append(ML_model.predict(data_test))
		#print(self.target_predict)
		self.label_predicted = np.transpose(np.array(self.target_predict))
		self.label_predicted = pd.DataFrame(self.label_predicted, columns=self.label_train.columns)
		
	def score_proposal(self, data_test, label_test):
		###############################################################################################################
		############################### Find the results for target ###################################################
		accuracy_per_target = np.zeros(len(label_test.columns))
		mean_squared_error_per_target = np.zeros(len(label_test.columns))
		for index, target in enumerate(label_test.columns):
			ML_model = self.ML_models_list[index]
			accuracy_per_target[index] = ML_model.score(data_test, label_test.loc[:,target])
			mean_squared_error_per_target[index] = mean_squared_error(label_test[target], self.label_predicted[target])
		###############################################################################################################

		###############################################################################################################
		############################### Find ML model results #########################################################
		
		accuracyOfModel = np.prod(accuracy_per_target)
		print('Accuracy of independently proposal is', accuracyOfModel)
		
		MSEmodel = np.sum(mean_squared_error_per_target)
		print('Mean Squared Error of independently proposal is', MSEmodel)
		###############################################################################################################

		return accuracyOfModel, MSEmodel, accuracy_per_target, mean_squared_error_per_target

# class Independently(HandleModel):
# 	"""docstring for multioutput"""
# 	def __init__(self, clf, data_train, label_train):
# 		super().__init__(clf, data_train, label_train)
	
# 	def train_model(self):
# 		self.clf.fit(self.data_train, self.label_train.loc[:,'azimuth']) 	

# 	def predict_model(self, data_test):
# 		self.label_predicted = self.clf.predict(data_test)
		
# 	def score_proposal(self, data_test, label_test):
# 		print(data_test)

# 		accuracyOfModel = self.clf.score(data_test, label_test.loc[:,'azimuth'])
# 		print('Accuracy of independently proposal is', accuracyOfModel)
		
# 		MSEmodel = np.sum(mean_squared_error_per_target)
# 		print('Mean Squared Error of independently proposal is', MSEmodel)
# 		###############################################################################################################

# 		return accuracyOfModel, MSEmodel, accuracy_per_target, mean_squared_error_per_target

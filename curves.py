import os
import numpy as np
import matplotlib.pyplot as plt
import natsort
from get_data import HandleData
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import SCORERS
# to handle Machine Learning model
from model import *

def scores(train_scores,test_scores):
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)
	return train_scores_mean, train_scores_std, test_scores_mean, test_scores_std

def plot_learning_curve(estimator, X, y, ylim=None, cv=None, n_jobs=4, train_sizes=np.linspace(.1, 1.0, 5)):
	plt.cla()
	train_sizes, train_scores, test_scores = \
		learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
					   # , return_times=True)
	train_scores_mean, train_scores_std, test_scores_mean, test_scores_std = scores(train_scores,test_scores)

	for index, value in enumerate(train_sizes):
		#print(index)
		print('train_scores:', train_scores_mean[index], 'test_scores', test_scores_mean[index], 'for {', value, '}')
	
	#plt.title(title)
	if ylim is not None:
		plt.ylim(*ylim)
	plt.xlabel("Training examples")
	plt.ylabel("Score")
	plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
	plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
	plt.legend(loc="best")
	
	fig_scatterPlot = folder_fig + '/' + 'learningCurve' + '/' + str(antenna_number) + '/'
	if not os.path.exists(fig_scatterPlot):
		os.makedirs(fig_scatterPlot)
	plt.savefig(fig_scatterPlot + 'learningCurve ' + str(antenna_number) + 'Ax' + str(SNR) + 'SNR' + '.png')

def plot_validation_curve(estimator, X, y, param_name, param_range, xlabel=None, ylim=None, n_jobs=None):
	# print(SCORERS.keys())
	plt.cla()
	train_scores, test_scores = validation_curve(
	estimator, X, y, param_name=param_name, param_range=param_range, scoring='neg_mean_squared_error', n_jobs=n_jobs)
	
	train_scores_mean, train_scores_std, test_scores_mean, test_scores_std = scores(train_scores,test_scores)

	#plt.title(title)
	# if ylim is not None:
	# 	plt.ylim(*ylim)
	# if xlabel is None:
	# 	xlabel = param_name
	# plt.xlabel(xlabel)
	# plt.ylabel("Score")
	lw = 2
	plt.semilogx(param_range, train_scores_mean, label="Training score",
				 color="darkorange", lw=lw)
	plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
				 color="navy", lw=lw)
	plt.legend(loc="best")
	
	fig_scatterPlot = folder_fig + '/' + 'validationCurve' + '/' + str(antenna_number) + '/'
	if not os.path.exists(fig_scatterPlot):
		os.makedirs(fig_scatterPlot)
	plt.savefig(fig_scatterPlot + 'validationCurve ' + str(antenna_number) + 'Ax' + str(SNR) + 'SNR' + '.png')


if __name__ == '__main__':
	#Type of antennas in the receiving system. Each type of antenna has its dataset
	antenna_type = 'dipoleVee'
	
	experiment = 'RegressionCorr_antenna_plus'

	data = 'data_corr_antenna_plus'
	# Folder where the dataset is for the receiving antennas defined in the variable antenna_type
	data_folder = './' + antenna_type + '/data/' + data

	folder_fig = './' + antenna_type + '/results_comparison/' + experiment + '/'
	if not os.path.exists(folder_fig):
		os.makedirs(folder_fig)
	
	# Percentage of the dataset that will be used to validate the ML models
	test_percentage = 0.2

	# Seed for same random results
	random_state = 42

	# Resolution of the proposal. If the model is going to solve the angles with a difference of 1 degrees, 2 degrees...
	# This variable will be used to obtain data from the dataset
	phase_list = [1]
	
	ML_model_dict = {'DT_Regressor' : DecisionTreeRegressor(random_state=random_state)}

	min_distance = 123
	distanceList = [213]
	
	list_proposal = [Multioutput]

	for phase in phase_list:
		print('\nphase angle: ' + str(phase))

		# For each ML model defined in ML_model_dict
		for ML_model_key in ML_model_dict:
			ML_model = ML_model_dict[ML_model_key]
			print('\n'+ ML_model_key + ' :' , ML_model)

			# Obtain the results of the proposal for different number of receiving antennas
			#for antenna_number in natsort.natsorted(os.listdir(data_folder)):
			for antenna_number in ['16']:
	
				print('\nnumber of antennas: ' + str(antenna_number))

				#################################################################################################################################################################################
				################################## Get the dataset for 'antenna_number' number of antennas ######################################################################################
				handleData = HandleData(phase, folder=data_folder + '/' + antenna_number, until='all')
				
				print('\nGetting data...')
				dataset = handleData.get_synthatic_data()
				#################################################################################################################################################################################
						
				for proposal in list_proposal:
					print(proposal.__name__)
					
					#################################################################################################################################################################################
					################################## MAIN #################################################################################################################################
					# Get the dataset to each SNR
					dataset_by_snr = dataset.groupby('SNR')
					
					
					for snr_by_snr in dataset_by_snr:
						SNR = snr_by_snr[0]
						
						# Get the power values obtained in the receiving antennas to train the ML model.
						power_data = snr_by_snr[1].loc[:, 'Pr0':]

						# Normalize power_data
						#power_data = power_data.div(power_data.sum(axis=1), axis=0)

						# Get the labels to train the ML model. In this case the labels will be made up of the azimuth and elevation angles
						label_data = snr_by_snr[1].loc[:, :'elevation']
						cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
						plot_learning_curve(ML_model, power_data, label_data, ylim=(0.7, 1.01), cv=cv, train_sizes=np.linspace(.1, 1.0, 10))
						#plot_validation_curve(ML_model, power_data, label_data, 'min_samples_split', np.linspace(1, 50, 10), xlabel=None, ylim=None, n_jobs=None)
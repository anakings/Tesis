import os
import natsort
from csv import reader
import matplotlib.pyplot as plt
#import glob
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import math 


def confusionMatrixPlot(total_df, label_name, step):
	dataset_by = total_df.groupby(['antenna number', 'SNR'])
	for labels_df in dataset_by:

		plt.cla()

		# Get the power values obtained in the receiving antennas to train the ML model.
		axis_x = labels_df[1].loc[:, label_name + ' validation']
		axis_y = labels_df[1].loc[:, label_name + ' predict']
		axis_y = axis_y.round(decimals=0)
		mat = confusion_matrix(axis_x, axis_y)

		classes = range(mat.shape[0])

		plt.figure(figsize=(25,15))
		sns.heatmap(mat.T[0::step,0::step], square=True, annot=True, fmt='d', cbar=False, xticklabels=classes[0::step], yticklabels=classes[0::step], cmap="Blues")
		plt.xlabel('Actual ' + label_name + ' angles')
		plt.ylabel('Estimated ' + label_name + ' angles')

		antenna_number = (labels_df[0])[0]
		SNR = (labels_df[0])[1]
		
		fig_save = folder_fig + '/' + 'confusionMatrix' + '/' + str(antenna_number) + '/'
		if not os.path.exists(fig_save):
			os.makedirs(fig_save)
		plt.savefig(fig_save + 'confusionMatrix ' + label_name + ' ' + str(antenna_number) + 'Ax' + str(SNR) + 'SNR' + '.png')

def scatterPlot(labels_df, label_name):
	dataset_by = labels_df.groupby(['antenna number', 'SNR'])
	for labels_df in dataset_by:
		plt.clf() 
		plt.cla()

		# Get the power values obtained in the receiving antennas to train the ML model.
		axis_x = labels_df[1].loc[:, label_name + ' validation']
		axis_y = labels_df[1].loc[:, label_name + ' predict']
		c = labels_df[1].iloc[0]['color graphs']
		plt.plot(axis_x, axis_y, color = c, marker='.', linestyle="None")
		plt.xlabel('Actual ' + label_name + ' angles')
		plt.ylabel('Estimated ' + label_name + ' angles')
			
		antenna_number = (labels_df[0])[0]
		SNR = (labels_df[0])[1]

		fig_save = folder_fig + '/' + 'scatterPlot' + '/' + str(antenna_number) + '/'
		if not os.path.exists(fig_save):
			os.makedirs(fig_save)
		plt.savefig(fig_save + 'The estimated ' + label_name + ' angles vs the actual ' + label_name + ' angles_' + str(SNR) + '.png')

def scatterPlot3D(labels_df, label_name):
	dataset_by_antenna = labels_df.groupby('antenna number')
	for df_by_antenna in dataset_by_antenna:
		plt.clf() 
		plt.cla()
		
		antenna_number	 = df_by_antenna[0]
		labels_df = df_by_antenna[1]

		c = labels_df.loc[:, 'color graphs']
		x = labels_df.loc[:, 'SNR']
		y = labels_df.loc[:, label_name + ' validation']
		z = labels_df.loc[:, label_name + ' predict']

		# Creating figure
		fig = plt.figure(figsize = (16, 9))
		ax = plt.axes(projection ="3d")
			   
		# Add x, y gridlines
		ax.grid(b = True, color ='grey',
		        linestyle ='-.', linewidth = 0.3,
		        alpha = 0.2)
			 
		# Creating plot
		sctt = ax.scatter3D(x, y, z,
		                    alpha = 0.8,
		                    c = c,
		                    marker ='.')
		 
		plt.title("simple 3D scatter plot")
		ax.set_xlabel('SNR', fontweight ='bold')
		ax.set_ylabel(label_name + ' validation', fontweight ='bold')
		ax.set_zlabel(label_name + ' predict', fontweight ='bold')

		fig_save = folder_fig + '/' + 'scatterPlot' + '/' + str(antenna_number) + '/'
		if not os.path.exists(fig_save):
			os.makedirs(fig_save)
		plt.savefig(fig_save + 'Scatter Plot 3D ' + label_name + ' ' + str(antenna_number) + ' Ax.png')		 

def lineChart(total_df, label_name, labels_df, x_label, y_label, y_min, y_max):
	plt.clf() 
	plt.cla()
	SMALL_SIZE = 8
	MEDIUM_SIZE = 10
	BIGGER_SIZE = 12

	plt.rc('lines', linewidth=2, color='r')
	plt.rc('font', size=BIGGER_SIZE)            # controls default text sizes
	plt.rc('axes', titlesize=BIGGER_SIZE)       # fontsize of the axes title
	plt.rc('axes', labelsize=BIGGER_SIZE)       # fontsize of the x and y labels
	plt.rc('xtick', labelsize=BIGGER_SIZE)      # fontsize of the tick labels
	plt.rc('ytick', labelsize=BIGGER_SIZE)      # fontsize of the tick labels
	plt.rc('legend', fontsize=BIGGER_SIZE)      # legend fontsize
	plt.rc('figure', titlesize=BIGGER_SIZE)     # fontsize of the figure title
	dataset_by_antenna = total_df.groupby('antenna number')
	for df_by_antenna in dataset_by_antenna:
		c = labels_df.loc[labels_df['antenna number'] == df_by_antenna[0], 'color graphs']
		axis_x = df_by_antenna[1].loc[:, 'SNR']
		axis_y = df_by_antenna[1].loc[:, label_name]
		label_list = str(df_by_antenna[0]) + ' Ax'
		# print(c)
		# print(axis_x)
		# print(axis_y)
		# print(label_list)
		plt.plot(axis_x, axis_y, color=c[0], label= label_list)
	plt.ylim(y_min, y_max)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.legend(loc = 'upper right')
	plt.grid(True)

	fig_save = folder_fig + '/' + 'lineChart' + '/'
	if not os.path.exists(fig_save):
		os.makedirs(fig_save)
	plt.savefig(fig_save + model + ' - ' +  label_name + ' - ' + y_label + ' vs ' + x_label + '.png')		
	
	#plt.savefig(folder_fig + model + ' - ' +  label_name + ' - ' + y_label + ' vs ' + x_label + '.png')

if __name__ == '__main__':
	experiment = 'RegressionCorr_antenna_plus'

	antenna_name = 'dipoleVee'
	model = 'DecisionTreeRegressor'
	label_name_list = ['azimuth','elevation']
	
	folder_results = './' + antenna_name + '/results/' + experiment + '/'
	folder_fig = './' + antenna_name + '/results_comparison/' + experiment + '/'
	if not os.path.exists(folder_fig):
		os.makedirs(folder_fig)

	
	logs_df = pd.DataFrame()
	labels_df = pd.DataFrame()

	for phase in os.listdir(folder_results):
		print(phase)

		phase_folder = folder_results + '/' + str(phase) + '/'

		for propose in os.listdir(phase_folder + '/'):
			print(propose)

			propose_folder = phase_folder + '/' + str(propose) + '/'

			folder_list = os.listdir(propose_folder)
			folder_list = [int(i) for i in folder_list]
			folder_list.sort()
	
			for antenna_number in folder_list:
			#for antenna_number in ['4', '6', '8', '10', '12']:

				antenna_folder = propose_folder + '/' + str(antenna_number) + '/'

				labels_temp = pd.read_csv(antenna_folder + model + '_labels.csv')
				labels_df = labels_df.append(labels_temp)
				
				logs_temp = pd.read_csv(antenna_folder + model + '.csv')
				logs_df = logs_df.append(logs_temp)
	
	lineChart(logs_df, 'mse', labels_df, 'SNR', 'MSE model', 0, 2000)
	
	logs_df['mse'] = logs_df['mse'].apply(lambda x: math.sqrt(x))
	lineChart(logs_df, 'mse', labels_df, 'SNR', 'RMSE model', 0, 45)

	lineChart(logs_df, 'prediction time', labels_df, 'SNR', 'Prediction time model', 0, 1)

	for label_name in label_name_list:
		lineChart_name = label_name + ' mse'
		lineChart_y_label = 'MSE model ' + label_name
		lineChart(logs_df, lineChart_name, labels_df, 'SNR', lineChart_y_label, 0, 2000)

		lineChart_y_label = 'R' + lineChart_y_label
		logs_df[lineChart_name] = logs_df[lineChart_name].apply(lambda x: math.sqrt(x))
		lineChart(logs_df, lineChart_name, labels_df, 'SNR', lineChart_y_label, 0, 45)

		scatterPlot3D(labels_df, label_name)
		scatterPlot(labels_df, label_name)
		confusionMatrixPlot(labels_df, label_name, 5)

	


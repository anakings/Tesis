import os
import natsort
from csv import reader
import matplotlib.pyplot as plt
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=BIGGER_SIZE)            # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)       # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)       # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)      # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)      # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)      # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)     # fontsize of the figure title

def confusionMatrixPlot(total_df, label_name, step):
	dataset_by = total_df.groupby(['antenna_number', 'SNR'])
	for labels_df in dataset_by:

		plt.cla()

		# Get the power values obtained in the receiving antennas to train the ML model.
		axis_x = labels_df[1].loc[:, label_name + '_validation']
		axis_y = labels_df[1].loc[:, label_name + '_predict']
		axis_y = axis_y.round(decimals=0)
		mat = confusion_matrix(axis_x, axis_y)

		classes = range(mat.shape[0])

		plt.figure(figsize=(25,15))
		sns.heatmap(mat.T[0::step,0::step], square=True, annot=True, fmt='d', cbar=False, xticklabels=classes[0::step], yticklabels=classes[0::step], cmap="Blues")
		plt.xlabel('Actual ' + label_name + ' angles')
		plt.ylabel('Estimated ' + label_name + ' angles')

		antenna_number = (labels_df[0])[0]
		SNR = (labels_df[0])[1]
		
		fig_scatterPlot = folder_fig + '/' + 'confusionMatrix' + '/' + str(antenna_number) + '/'
		if not os.path.exists(fig_scatterPlot):
			os.makedirs(fig_scatterPlot)
		plt.savefig(fig_scatterPlot + 'confusionMatrix ' + label_name + ' ' + str(antenna_number) + 'Ax' + str(SNR) + 'SNR' + '.png')

def scatterPlot(labels_df, label_name):
	dataset_by = labels_df.groupby(['antenna_number', 'SNR'])
	for labels_df in dataset_by:
		plt.clf() 
		plt.cla()

		# Get the power values obtained in the receiving antennas to train the ML model.
		axis_x = labels_df[1].loc[:, label_name + '_validation']
		axis_y = labels_df[1].loc[:, label_name + '_predict']
		c = labels_df[1].iloc[0]['color_graphs_3D']
		plt.plot(axis_x, axis_y, color = c, marker='.', linestyle="None")
		plt.xlabel('Actual ' + label_name + ' angles')
		plt.ylabel('Estimated ' + label_name + ' angles')
			
		antenna_number = (labels_df[0])[0]
		SNR = (labels_df[0])[1]

		fig_scatterPlot = folder_fig + '/' + 'scatterPlot' + '/' + str(antenna_number) + '/'
		if not os.path.exists(fig_scatterPlot):
			os.makedirs(fig_scatterPlot)
		plt.savefig(fig_scatterPlot + 'The estimated ' + label_name + ' angles vs the actual ' + label_name + ' angles_' + str(SNR) + '.png')

def scatterPlot3D(labels_df, label_name):
	dataset_by_antenna = labels_df.groupby('antenna_number')
	for df_by_antenna in dataset_by_antenna:
		plt.clf() 
		plt.cla()
		antenna_number	 = df_by_antenna[0]
		labels_df = df_by_antenna[1]

		c = labels_df.loc[:, 'color_graphs_3D']
		x = labels_df.loc[:, 'SNR']
		y = labels_df.loc[:, label_name + '_validation']
		z = labels_df.loc[:, label_name + '_predict']

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

		fig_scatterPlot = folder_fig + '/' + 'scatterPlot' + '/' + str(antenna_number) + '/'
		if not os.path.exists(fig_scatterPlot):
			os.makedirs(fig_scatterPlot)
		plt.savefig(fig_scatterPlot + 'Scatter Plot 3D ' + label_name + ' ' + str(antenna_number) + ' Ax.png')		 

def lineChart(total_df, labels_df, x_label, y_label, y_min, y_max):
	plt.clf() 
	plt.cla()
	dataset_by_antenna = total_df.groupby('antenna_number')
	for df_by_antenna in dataset_by_antenna:
		print(df_by_antenna[0])
		c = labels_df.loc[labels_df['antenna_number'] == df_by_antenna[0], 'color_graphs_3D']
		axis_x = df_by_antenna[1].loc[:, 'SNR']
		axis_y = df_by_antenna[1].loc[:, 'mse']
		label_list = str(df_by_antenna[0]) + ' Ax'
		plt.plot(axis_x, axis_y, color=c[0], label= label_list)
	plt.ylim(y_min, y_max)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.legend(loc = 'upper right')
	plt.grid(True)
	plt.savefig(folder_fig + model + ' - ' + y_label + ' vs ' + x_label + '.png')
	plt.show()

if __name__ == '__main__':
	experiment = 'RegressionCov'

	antenna_name = 'dipoleVee'
	model = 'DecisionTreeRegressor'
	#model = 'DecisionTreeRegressor'
	
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
	
			#for antenna_number in folder_list:
			for antenna_number in ['4','6','8','10','12','14','16']:

				antenna_folder = propose_folder + '/' + str(antenna_number) + '/'

				labels_temp = pd.read_csv(antenna_folder + model + '_labels.csv')
				labels_df = labels_df.append(labels_temp)
				
				logs_temp = pd.read_csv(antenna_folder + model + '.csv')
				logs_df = logs_df.append(logs_temp)
	

	# lineChart(logs_df, labels_df, 'SNR', 'MSE', 0, 2000)

	# label_name = 'azimuth'
	# scatterPlot3D(labels_df, label_name)
	# scatterPlot(labels_df, label_name)
	# # # confusionMatrixPlot(labels_df, label_name, 5)

	# label_name = 'elevation'
	# scatterPlot3D(labels_df, label_name)
	# scatterPlot(labels_df, label_name)
	# # # confusionMatrixPlot(labels_df, label_name, 1)

	import math 
	logs_df['mse'] = logs_df['mse'].apply(lambda x: math.sqrt(x))
	lineChart(logs_df, labels_df, 'SNR', 'RMSE', 0, 200)

	

	


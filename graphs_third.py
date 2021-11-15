import os
from natsort import natsorted
from csv import reader
import matplotlib.pyplot as plt
import numpy as np

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

if __name__ == '__main__':
	if 'Results' not in os.listdir('./'): os.mkdir('./Results')
	#folder_results = './GridSearch/Logs/'
	folder_results = './Logs/'
	text = 'Accuracy'
	propose = 'third_proposal'


	for phase in natsorted(os.listdir(folder_results)):

		folder_results_list = natsorted(os.listdir(folder_results + str(phase) + '/'))
	
		matrix_y = np.zeros((len(os.listdir(folder_results + str(phase) + '/' + folder_results_list[0] + '/')),len(natsorted(os.listdir(folder_results + str(phase) + '/')))))
		axis_x = []
		antenna_number = []

		for count_distance, distance in enumerate(folder_results_list):
			#distance_list = natsorted(os.listdir(folder_results + str(phase) + '/'))
			#print(distance)
			axis_x.append(int(distance))
			for count_antenna, antenna in enumerate(natsorted(os.listdir(folder_results + str(phase) + '/' + str(distance) + '/'))):
				antenna_number.append(antenna) #for the legend
				print(antenna)
				for model in os.listdir(folder_results + str(phase) + '/' + str(distance) + '/' + str(antenna) + '/' + str(propose) + '/'):
						
					#read to file './Logs/'+ str(i) + '/' + model: j and the accuracy
					with open(folder_results + str(phase) + '/' + str(distance) + '/' + str(antenna) + '/' + str(propose) + '/' + str(model), 'rt') as f:
						data = reader(f)
						for row in data:
							matrix_y[count_antenna,count_distance] = float(row[1])

		for count, value in enumerate(matrix_y):
			ploteo = plt.plot(axis_x,value,label=antenna_number[count])
		plt.title(f'{text} vs distance (resolution {phase}°)')
		plt.xlabel('distances')
		plt.ylabel(text)
		plt.ylim(80,100)
		#plt.yscale("log")
		plt.legend(loc='best', shadow=True)
		plt.grid(True)
		plt.savefig('Results/' + 'accuracy vs distance_' + str(phase) + '_' + propose + '.png')
		plt.show()

				
		
		

import os
import natsort
from csv import reader
import matplotlib.pyplot as plt

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

def showFig(text, listToPlot_first):
	#make graph with axis_x and axis_y
	plt.figure(figsize=(8, 8))
	index = 0
	for propose in os.listdir(folder_results):
		for i in os.listdir(folder_results + str(propose) + '/'):	
			for name in os.listdir(folder_results + str(propose) + '/' + str(i) + '/'):
				print(index)
				plt.plot([j for j in axis_x], listToPlot_first[index], '-d', label=name.split('.')[0] + '_' + str(propose))
				index += 1

	# plt.yscale('log')
	plt.title(text + ' vs ' + str(i))
	plt.xlabel(str(i))
	plt.ylabel(text)
	plt.legend(loc='best', shadow=True)
	plt.grid(True)
	plt.savefig('Results/' + text + '_' + str(i) + '.png')
	plt.show()

if __name__ == '__main__':
	if 'Results' not in os.listdir('./'): os.mkdir('./Results')
	#folder_results = './GridSearch/Logs/'
	folder_results = './Logs/'
	axis_y_first = []
	axis_y_first_timeVal = []

	for propose in os.listdir(folder_results):
		for i in os.listdir(folder_results + str(propose) + '/'):	
			for model in os.listdir(folder_results + str(propose) + '/' + str(i) + '/'):
				accuracyList = []
				#timeValidationList = []
				axis_x = []
				
				#read to file './Logs/'+ str(i) + '/' + model: j and the accuracy
				with open(folder_results + str(propose) + '/' + str(i) + '/' + str(model), 'rt') as f:
					data = reader(f)
					for row in data:
						axis_x.append(int(row[0]))
						accuracyList.append(float(row[1]))
						#timeValidationList.append(float(row[2]))

				axis_y_first.append(accuracyList)
			#axis_y_first_timeVal.append(timeValidationList)
	print(len(axis_y_first))

	figAccurracy = showFig('Accurracy', axis_y_first)
		
		

import os
import numpy as np
import matplotlib.pyplot as plt
import natsort
from get_data import HandleData
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier

def scores(train_scores,test_scores):
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    return train_scores_mean, train_scores_std, test_scores_mean, test_scores_std

def plot_validation_curve(estimator, title, X, y, param_name, param_range, xlabel=None, ylim=None, n_jobs=None):
	
	train_scores, test_scores = validation_curve(
    estimator, X, y, param_name=param_name, param_range=param_range,
    scoring="accuracy", n_jobs=n_jobs)
	
	train_scores_mean, train_scores_std, test_scores_mean, test_scores_std = scores(train_scores,test_scores)

	plt.title(title)
	if ylim is not None:
		plt.ylim(*ylim)
	if xlabel is None:
		xlabel = param_name
	plt.xlabel(xlabel)
	plt.ylabel("Score")
	lw = 2
	plt.semilogx(param_range, train_scores_mean, label="Training score",
    	         color="darkorange", lw=lw)
	plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
    	         color="navy", lw=lw)
	plt.legend(loc="best")
	plt.savefig('./images resul/' + title + '.png', dpi=600)
	plt.show()
    #plt.close()

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=4, train_sizes=np.linspace(.1, 1.0, 5), savefig=None):

    train_sizes, train_scores, test_scores = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,shuffle=True)
                       # , return_times=True)
    train_scores_mean, train_scores_std, test_scores_mean, test_scores_std = scores(train_scores,test_scores)

    for index, value in enumerate(train_sizes):
    	#print(index)
    	print('train_scores:', train_scores_mean[index], 'test_scores', test_scores_mean[index], 'for {', value, '}')
    
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    if savefig == None:
    	savefig = title
    plt.savefig('./Results/learning_curve/' + savefig + '_' + str(phase) + '_' + str(distanceValue) + '_' + str(j) +'.png', dpi=600)
    plt.show()

if __name__ == '__main__':
    
    Data_Folder = './DOA_Data' #Folder where the data is
  
    iter_distance_data = 50
    samples_distance= 50
    
    modelList = [DecisionTreeClassifier(max_depth=300, random_state=42)]
    modelName = ['DT']

    min_distance = 10
    max_distance_range = 300
    min_distance_range = 100
    distanceList = np.linspace(min_distance_range, max_distance_range, num = 2, dtype = int)

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

                    handleData = HandleData(phase, folder = Data_Folder + '/' + os.listdir(Data_Folder)[0] + '/' + str(j), iterations = iter_distance_data, until=number_files, samples_by_distance=samples_distance) #folder = './Dround_Data_New/Nomalized'
                        
                    print('Getting data...')
                    antenna_data, label_list, label_matrix = handleData.get_synthatic_data() #label_list: for the second proposal, label_matrix: for the first and third proposal 

                    #plot Learning Curve
                    title = r"Learning Curves (modelName[name])"
                    cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
                    estimator = MultiOutputClassifier(model)
                    plot_learning_curve(estimator, title, antenna_data, label_matrix, ylim=(0.7, 1.01), cv=cv, train_sizes=np.linspace(.1, 1.0, 5), savefig=modelName[name])
       
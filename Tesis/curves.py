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

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=4, train_sizes=np.linspace(.1, 1.0, 5), savefig=None):

    train_sizes, train_scores, test_scores = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
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
    plt.savefig('./Results/learning_curve/' + savefig + '.png', dpi=600)
    plt.show()

if __name__ == '__main__':

    Data_Folder = './DOA_Data'
    
    test_percentage = 0.2
    phase = 5

    '''
    modelList = [DecisionTreeClassifier(random_state=42), BaggingClassifier(DecisionTreeClassifier(random_state=42), bootstrap=False, n_jobs=-1, random_state=42), SVC()]
    modelName = ['Decision Tree Classifier', 'Bagging Classifier', SVC]
    param_grid_list = [[{'max_depth': [x for x in range(1,25)], 'min_samples_split': np.linspace(0.1, 1.0, 10)},
    {'max_depth': [x for x in range(1,25)]}],[{'n_estimators': [x for x in range(100,1100,100)], 'max_samples': np.linspace(0.1, 1.0, 5)}],[{'kernel': ['rbf','poly','sigmoid'], 'gamma': np.logspace(-6, 10, 5), 'C': [1, 10, 100, 1000,10000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000,10000]}]]

    modelList = [RandomForestClassifier(n_estimators=10)]
    modelName = ['RF']
    
    
    '''

    modelList = [DecisionTreeClassifier()]
    modelName = ['DecisionTreeClassifier']
 
    
    for name,model in enumerate(modelList):

        dirFiles = os.listdir(Data_Folder + '/' + 'antennas')
        sortedDirFiles = natsort.natsorted(dirFiles)
                
        handleData = HandleData(phase, folder = Data_Folder + '/' + 'antennas' + '/' + dirFiles[1], iterations = 'all') #folder = './Dround_Data_New/Nomalized'
                
        antenna_data, label_list, label_matrix = handleData.get_synthatic_data()

        #plot Learning Curve
        title = r"Learning Curves (modelName[name])"
        cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
        estimator = MultiOutputClassifier(model)
        plot_learning_curve(estimator, title, antenna_data, label_matrix, ylim=(0.7, 1.01), cv=cv, train_sizes=np.linspace(.1, 1.0, 10), savefig=modelName[name])
       
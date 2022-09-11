import optuna
import sklearn.ensemble
import sklearn.model_selection
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

class Objective(object):
    def __init__(self, x_train, x_val, y_train, y_val):
        self.x_train, self.x_val, self.y_train, self.y_val = x_train, x_val, y_train, y_val

    def __call__(self, trial):
        '''
        rf_max_depth = int(trial.suggest_loguniform('rf_max_depth', 2, 32))
        rf_n_estimators = int(trial.suggest_loguniform('rf_n_estimators', 5, 20))
        classifier_obj = MultiOutputClassifier(sklearn.ensemble.RandomForestClassifier(
            max_depth=rf_max_depth, n_estimators=rf_n_estimators))
        '''
        # CLASSIFICATION
        # rf_criterion = trial.suggest_categorical('rf_criterion', ['gini', 'entropy'])
        # rf_splitter = trial.suggest_categorical('rf_splitter', ['best', 'random'])
        # rf_max_depth = int(trial.suggest_int('rf_max_depth', 100, 1100, step = 100))
        # rf_min_samples_split = int(trial.suggest_int('rf_min_samples_split', 2, 40))
        # rf_min_samples_leaf = int(trial.suggest_int('rf_min_samples_leaf', 1, 40))
        # # rf_min_weight_fraction_leaf = trial.suggest_float('rf_min_weight_fraction_leaf', 0.1, 0.5, step=0.1)
        # rf_max_features = trial.suggest_categorical('rf_max_features', ['auto', 'sqrt', 'log2'])

        # REGRESSION
        rf_criterion = trial.suggest_categorical('rf_criterion', ['friedman_mse', 'poisson'])
        rf_splitter = trial.suggest_categorical('rf_splitter', ['best', 'random'])
        rf_max_depth = int(trial.suggest_int('rf_max_depth', 100, 1100, step = 100))
        rf_min_samples_split = int(trial.suggest_int('rf_min_samples_split', 2, 40))
        rf_min_samples_leaf = int(trial.suggest_int('rf_min_samples_leaf', 1, 40))
        # rf_min_weight_fraction_leaf = trial.suggest_float('rf_min_weight_fraction_leaf', 0.1, 0.5, step=0.1)
        rf_max_features = trial.suggest_categorical('rf_max_features', ['auto', 'sqrt', 'log2'])
        
        classifier_obj = MultiOutputRegressor(
            DecisionTreeRegressor(
                criterion=rf_criterion, 
                splitter=rf_splitter, 
                max_depth=rf_max_depth, 
                min_samples_split=rf_min_samples_split, 
                min_samples_leaf=rf_min_samples_leaf
            )
        )

        classifier_obj.fit(self.x_train, self.y_train)

        y_pred = classifier_obj.predict(self.x_val)
       
        mean_squared_error_per_target = []
        for i in range(y_pred.shape[1]):
            label_pred_target = y_pred[:,i]
            label_validation_target = self.y_val.to_numpy()[:,i]
                                
            mean_squared_error_per_target.append(mean_squared_error(label_validation_target, y_pred[:,i]))

        print('\nLABEL VALIDATION:')
        print(self.y_val)

        print('\nLABEL PREDICTED:')
        print(y_pred)

        print('\nMSE PER TARGET:', mean_squared_error_per_target)

        mse = sum(mean_squared_error_per_target)
        print('MSE:', mse)
        
        return mse

def optuna_main(x_train, x_val, y_train, y_val):
    objective = Objective(x_train, x_val, y_train, y_val)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=5)

    fig = optuna.visualization.plot_param_importances(study)
    fig.show()

    return  study.best_value, study.best_params, study.best_trial

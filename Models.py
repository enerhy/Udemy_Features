# Random Forest
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

def run_RandomForest(X_train, X_test, ytrain, ytest):
    regressor_rf = RandomForestRegressor(max_depth=15, min_samples_split=15, random_state=0)
    regressor_rf.fit(X_train, ytrain)
    
    #Making prediction:
    X_train_preds = regressor_rf.predict(X_train)
    X_test_preds = regressor_rf.predict(X_test)
    
        #Printing the results:
    print('train mse: {}'.format(mean_squared_error(ytrain, X_train_preds)))
    print('train rmse: {}'.format(sqrt(mean_squared_error(ytrain, X_train_preds))))
    print('train r2: {}'.format(r2_score(ytrain, X_train_preds)))
    print()
    print('test mse: {}'.format(mean_squared_error(ytest, X_test_preds)))
    print('test rmse: {}'.format(sqrt(mean_squared_error(ytest, X_test_preds))))
    print('test r2: {}'.format(r2_score(ytest, X_test_preds)))
    
-----GRIDS SEARCH Function---
 # Random Forest
def grid_search_RF(Xtrain, ytrain):
    from sklearn.model_selection import GridSearchCV
    #first we specifiy the parameter to test, than we create an object that takes the parameter as input
    #where one dictionary is one of the models to testm where we include the different parameters to test as lists
    rf_regressor = RandomForestRegressor()
    parameters = [{'n_estimators' : [1, 10, 100, 50], 
                   'max_depth' : [15, 20, 10],
                   'min_samples_split' : [10, 20, 40],
                   'min_samples_leaf' : [2, 5]
                      }]
    gridsearch = GridSearchCV(estimator = rf_regressor,
                              param_grid = parameters,
                              scoring = 'r2',
                              cv = 3,
                              n_jobs=-1
                              )
    
    gridsearch.fit(Xtrain, ytrain)
    
    return gridsearch.best_params_, gridsearch.best_score_, gridsearch.best_estimator_
    
    

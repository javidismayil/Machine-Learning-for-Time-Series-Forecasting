import pandas as pd
data = pd.read_excel('D:/JAVID_ISMAYILOV_DERS/2.donem/machine-learning/vize/final/EVDS.xlsx')
lag_data1=data['TP N2SY01 2005'].shift(1)
lag_data2=data['TP N2SY01 2005'].shift(2)
lag_data3=data['TP N2SY01 2005'].shift(3)
################
data_with_lag = pd.concat([data, lag_data1, lag_data2, lag_data3], axis=1)
data_with_lag.columns = list(data.columns) + ['lag1', 'lag2', 'lag3']
data_with_lag = data_with_lag.dropna()
y=data_with_lag.iloc[:,1]
X=data_with_lag.iloc[:,2:]
#############
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, random_state=0,
                                                   test_size=0.2)
X_train=X[0:72]
y_train=y[0:72]
X_test=X[73:93]
y_test=y[73:93]

##############SVR
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

svr=SVR()
param_range = [0.1, 1.0, 10.0]
param_grid = {  'C': param_range}
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'kernel': ['linear', 'rbf'],
              'gamma': ['scale', 'auto']}
gs_SVR = GridSearchCV(estimator=svr, param_grid=param_grid,
                  scoring='neg_mean_squared_error',cv=2,refit=True, n_jobs=-1)
gs_SVR = gs_SVR.fit(X_train, y_train)
print(gs_SVR.best_score_)
print(gs_SVR.best_params_)
best_svr=gs_SVR.best_estimator_
y_pred_svr = best_svr.predict(X_test)
mse_svr = mean_squared_error(y_test, y_pred_svr)
print("Mean Squared Error: ", mse_svr)


############################
from sklearn.tree import DecisionTreeRegressor
decison_tree = DecisionTreeRegressor()
param_grid = {'max_depth': [None, 5, 10, 15],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4],
              'max_features': ['auto', 'sqrt', 'log2']}
gs_tree = GridSearchCV(decison_tree, param_grid, cv=5, scoring='neg_mean_squared_error')
gs_tree.fit(X_train, y_train)
best_tree = gs_tree.best_estimator_
y_pred_tree = best_tree.predict(X_test)
mse_tree = mean_squared_error(y_test, y_pred_tree)
print("Mean Squared Error: ", mse_tree)
print(best_tree)
###################
from sklearn.ensemble import RandomForestRegressor
random_forest = RandomForestRegressor()
param_grid = {'n_estimators': [10, 15, 20],
              'max_depth': [None, 5, 10, 15],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4],
              'max_features': ['auto', 'sqrt', 'log2']}
gs_forest = GridSearchCV(random_forest, param_grid, cv=5, scoring='neg_mean_squared_error')
gs_forest.fit(X_train, y_train)
best_forest = gs_forest.best_estimator_
y_pred_forest = best_forest.predict(X_test)
mse_forest = mean_squared_error(y_test, y_pred_forest)
print(best_forest)
print(mse_forest)
#################
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor()
param_grid = {'n_neighbors': [3, 5, 7, 10],
              'weights': ['uniform', 'distance'],
              'algorithm': ['auto', 'ball_tree', 'kd_tree'],
              'leaf_size': [3, 5, 10, 20]}

gs_knn = GridSearchCV(knn, param_grid, cv=5, scoring='neg_mean_squared_error')
gs_knn.fit(X_train, y_train)
best_knn = gs_knn.best_estimator_
print("Best Parameters: ", gs_knn.best_params_)
y_pred_knn = best_knn.predict(X_test)

mse_knn = mean_squared_error(y_test, y_pred_knn)
print(best_knn)
print("Mean Squared Error: ", mse_knn)

#######################regression
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
gs_reg = GridSearchCV(reg, param_grid, cv=5, scoring='neg_mean_squared_error')
gs_reg.fit(X_train, y_train)
best_reg = gs_reg.best_estimator_
print("Best Parameters: ", gs_reg.best_params_) 
y_pred_reg = best_reg.predict(X_test)
mse_reg = mean_squared_error(y_test, y_pred_reg)
print("Mean Squared Error: ", mse_reg)
reg.fit(X_train, y_train)
y_pred=reg.predict(X_train)
mse_reg = mean_squared_error(y_test, y_pred)
mse_reg


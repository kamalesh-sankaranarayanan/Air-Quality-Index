# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 18:17:25 2021

@author: Harish
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb


df = pd.read_csv('C:/Users/Harish/Documents/Projects/Air Quality Index/Data/Real-Data/Real_Combine.csv')
df = df.dropna(axis = 0)

df
X = df.drop('PM 2.5', axis=1) #independent feature
y = df['PM 2.5'] #dependent feature


sns.pairplot(df)

df.corr()

corrmat = df.corr()
tf = corrmat.index
sns.heatmap(df[tf].corr(), annot=True)


#Feature Importance

model = ExtraTreesRegressor()

model.fit(X,y)


fi = pd.Series(model.feature_importances_, index=X.columns)
fi.nlargest(5).plot(kind='barh')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# y = mx + c
reg = LinearRegression()
reg.fit(X_train, y_train)

print('Coefficient of determination R^2 on train set: {}'.format(reg.score(X_train, y_train)))
print('Coefficient of determination R^2 on test set: {}'.format(reg.score(X_test, y_test)))


score = cross_val_score(reg, X, y, cv=5)

score.mean()


#Model Evaluation

coeff_df = pd.DataFrame(reg.coef_, X.columns, columns=['Coefficient'])
coeff_df
"""Holding all other features fixed, a unit increaces in select column eg.'T' is associated with an 
inrcease (if coeff is +ve) or decrease (if coeff is -ve) of coefficient value in dependented feature"""



pred = reg.predict(X_test)
sns.distplot(y_test - pred)
plt.scatter(y_test, pred)
#should be a gaussian cure (bell cure) for the best model


print('MAE: ', metrics.mean_absolute_error(y_test, pred))
print('MSE: ', metrics.mean_squared_error(y_test, pred))
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test, pred)))



file = open('Regression.pkl', 'wb')

pickle.dump(reg, file)

#comparing Linear, LASSO, Ridge

lin_reg = LinearRegression()
mse = cross_val_score(lin_reg, X, y, scoring='neg_mean_squared_error', cv=5)
mean_mse = np.mean(mse)
mean_mse



ridge = Ridge()
parameters = {'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40]}
ridge_reg = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=5)
ridge_reg.fit(X,y)

print(ridge_reg.best_params_)
print(ridge_reg.best_score_)


lasso = Lasso()
parameters = {'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40]}
lasso_reg = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv=5)
lasso_reg.fit(X,y)

print(lasso_reg.best_params_)
print(lasso_reg.best_score_)

prediction = lasso_reg.predict(X_test)
sns.distplot(y_test-prediction)
plt.scatter(y_test,prediction)


print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))



file = open('lasso_regression_model.pkl', 'wb')

# dump information to that file
pickle.dump(lasso_reg, file)



# Random forest
regressor=RandomForestRegressor()
regressor.fit(X_train,y_train)
score = cross_val_score(regressor,X,y,cv=5)
score.mean()

prediction=regressor.predict(X_test)

#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]

# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]

# Method of selecting samples for training each tree
# bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = 1)
rf_random.fit(X_train,y_train)

rf_random.best_params_

rf_random.best_score_

predictions=rf_random.predict(X_test)

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# open a file, where you ant to store the data
file = open('random_forest_regression_model.pkl', 'wb')

# dump information to that file
pickle.dump(rf_random, file)


#XGboost
regressor = xgb.XGBRegressor()
regressor.fit(X_train,y_train)

score = cross_val_score(regressor,X,y,cv=5)
score.mean()
prediction=regressor.predict(X_test)

#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Various learning rate parameters
learning_rate = ['0.05','0.1', '0.2','0.3','0.5','0.6']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
#Subssample parameter values
subsample=[0.7,0.6,0.8]
# Minimum child weight parameters
min_child_weight=[3,4,5,6,7]


# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'learning_rate': learning_rate,
               'max_depth': max_depth,
               'subsample': subsample,
               'min_child_weight': min_child_weight}

print(random_grid)

regressor = xgb.XGBRegressor()

xg_random = RandomizedSearchCV(estimator = regressor, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = 1)
xg_random.fit(X_train,y_train)
xg_random.best_params_
xg_random.best_score_
predictions=xg_random.predict(X_test)
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))













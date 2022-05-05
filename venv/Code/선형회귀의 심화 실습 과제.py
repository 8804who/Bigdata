from sklearn import linear_model
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data_url = 'D:/공부/수업 자료/4-1/USG공유대학/빅데이터응용/실습자료/housing.data'
df_data=pd.read_csv(data_url,sep='\s+', header=None)
df_data.columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']


y_data=df_data["MEDV"]
y_data=y_data.values
y_data=y_data.reshape(-1,1)

df_data=df_data.drop("MEDV",axis=1)
x_data=df_data.values

minmax_scale=preprocessing.MinMaxScaler(feature_range=(0,5)).fit(x_data)
x_scaled_data = minmax_scale.transform(x_data)

X_train, X_test, y_train, y_test = train_test_split(x_scaled_data,y_data,test_size=0.33)


regr=linear_model.LinearRegression(fit_intercept=True, copy_X=True, n_jobs=8)
lasso_regr=linear_model.Lasso(alpha=0.01, fit_intercept=True, copy_X=True)
ridger_regr=linear_model.Ridge(alpha=0.01, fit_intercept=True, copy_X=True)
SGD_regr=linear_model.SGDRegressor(penalty="l2",alpha=0.01,max_iter=1000,tol=0.001,eta0=0.3)

regr.fit(X_train, y_train)
lasso_regr.fit(X_train, y_train)
ridger_regr.fit(X_train, y_train)
SGD_regr.fit(X_train, y_train)

y_true = y_test.copy()
regr_y_hat = regr.predict(X_test)
lasso_y_hat = lasso_regr.predict(X_test)
ridger_y_hat = ridger_regr.predict(X_test)
SGD_y_hat = SGD_regr.predict(X_test)

print("regr:", r2_score(y_true, regr_y_hat), mean_absolute_error(y_true,regr_y_hat), mean_squared_error(y_true, regr_y_hat))
print("lasso:", r2_score(y_true, lasso_y_hat), mean_absolute_error(y_true,lasso_y_hat), mean_squared_error(y_true, lasso_y_hat))
print("ridger:", r2_score(y_true, ridger_y_hat), mean_absolute_error(y_true,ridger_y_hat), mean_squared_error(y_true, ridger_y_hat))
print("SGD:", r2_score(y_true, SGD_y_hat), mean_absolute_error(y_true,SGD_y_hat), mean_squared_error(y_true, SGD_y_hat))


plt.subplot(2,2,1)
plt.scatter(y_true, regr_y_hat, s=10)
plt.xlabel("Prices: $Y_i $")
plt.ylabel("Predicted prices: $What {Y}_i $")
plt.title("regr")

plt.subplot(2,2,2)
plt.scatter(y_true, lasso_y_hat, s=10)
plt.xlabel("Prices: $Y_i $")
plt.ylabel("Predicted prices: $What {Y}_i $")
plt.title("lasso")

plt.subplot(2,2,3)
plt.scatter(y_true, ridger_y_hat, s=10)
plt.xlabel("Prices: $Y_i $")
plt.ylabel("Predicted prices: $What {Y}_i $")
plt.title("ridger")

plt.subplot(2,2,4)
plt.scatter(y_true, SGD_y_hat, s=10)
plt.xlabel("Prices: $Y_i $")
plt.ylabel("Predicted prices: $What {Y}_i $")
plt.title("SGD")

plt.show()
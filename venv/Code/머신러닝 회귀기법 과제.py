import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

x=np.arange(10).reshape(-1,1)
y=np.array([0,0,0,0,1,1,1,1,1,1])

regr = linear_model.LinearRegression(fit_intercept=True).fit(x,y)
regr_nointercept = linear_model.LinearRegression(fit_intercept=False).fit(x,y)
logreg = LogisticRegression(fit_intercept=True).fit(x,y)

regr_y_hat = regr.predict(x)
print("regr:", r2_score(y, regr_y_hat), mean_absolute_error(y,regr_y_hat), mean_squared_error(y, regr_y_hat))

regr_nointercept_y_hat = regr_nointercept.predict(x)
print("regr_nointercept:", r2_score(y, regr_nointercept_y_hat), mean_absolute_error(y,regr_nointercept_y_hat), mean_squared_error(y, regr_nointercept_y_hat))

x_test = np.linspace(0,10,300)
loss=expit(x_test*logreg.coef_+logreg.intercept_).ravel()
print(confusion_matrix(y, logreg.predict(x)))
print(accuracy_score(y, logreg.predict(x)))
print(precision_score(y, logreg.predict(x)))
print(recall_score(y, logreg.predict(x)))
print(f1_score(y, logreg.predict(x)))

plt.scatter(x,y)
plt.plot(x, regr.coef_*x+regr.intercept_, 'g')
plt.plot(x, regr_nointercept.coef_*x+regr_nointercept.intercept_,'b')
plt.plot(x_test, loss, color="red", linewidth=3)
plt.show()
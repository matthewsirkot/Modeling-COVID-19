import math
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import seed
seed(1)
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import tensorflow
tensorflow.random.set_seed(1)
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score

#import covid test data
#contains number of covid tests given per date, for the entire state
cTests = pd.read_csv('COVID-19_PCR_Test_Counts_March_2020_-_Current_Statewide_Health.csv', usecols=['Date', 'New PCR Tests'],  parse_dates=['Date'], index_col=[0])
cTests = cTests.sort_values(by=["Date"], ascending=True)
plt.show()

#import covid case count data
#contains daily number of new cases per date, by county
cCases = pd.read_csv('COVID-19_Aggregate_Cases_Current_Daily_County_Health.csv', usecols=['Date', 'New Cases'],  parse_dates=['Date'], index_col=[0])
cCases = cCases.sort_values(by=["Date"], ascending=True)
cCases = cCases.groupby('Date').sum()
plt.plot(cCases)
plt.show()

#import covid death data
#contains daily number of new deaths per date, by county
cDeaths = pd.read_csv('COVID-19_Aggregate_Death_Data_Current_Daily_County_Health.csv', usecols=['Date of Death', 'New Deaths'],  parse_dates=['Date of Death'], index_col=[0])
cDeaths = cDeaths.sort_values(by=["Date of Death"], ascending=True)
cDeaths = cDeaths.groupby('Date of Death').sum()
plt.plot(cDeaths)
plt.show()

#inner join of datasets to ensure data agrees on date. This was necessary
#to combine multiple case and death entries from various counties into a single date.
#also ensures tuples agree on date.
data = cTests.join(cCases).join(cDeaths, how="inner")


#features: tests given and new cases recorded
X = data[['New PCR Tests', 'New Cases']]

#target: daily number of deaths
Y = data[['New Deaths']]

#split data into training and validation data
X_train, X_val, y_train, y_val = train_test_split(X, Y)

y_train = np.reshape(y_train, (-1,1))
y_val = np.reshape(y_val, (-1,1))

#Scale the data
from sklearn import preprocessing
scaler_x = preprocessing.MinMaxScaler()
scaler_y = preprocessing.MinMaxScaler()

print(scaler_x.fit(X_train))
xtrain_scale=scaler_x.transform(X_train)
print(scaler_x.fit(X_val))
xval_scale=scaler_x.transform(X_val)
print(scaler_y.fit(y_train))
ytrain_scale=scaler_y.transform(y_train)
print(scaler_y.fit(y_val))
yval_scale=scaler_y.transform(y_val)
print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

model = LinearRegression()
scores = []
model.fit(xtrain_scale, ytrain_scale)
ypred = model.predict(xval_scale)

print('Coefficients: \n', model.coef_)
print('Mean squared error: %.2f'% mean_squared_error(yval_scale, ypred))
print('Coefficient of determination: %.2f'% r2_score(yval_scale, ypred))


predictions = scaler_y.inverse_transform(ypred)

print(mean_absolute_error(y_val, predictions))
print(mean_squared_error(y_val, predictions))
print(math.sqrt(mean_squared_error(y_val, predictions)))
print(np.mean(predictions))

plt.plot(predictions, 'ro')
plt.ylabel('deaths')
plt.xlabel('time')
plt.suptitle('Predicted (Linear Regression)')
plt.show()









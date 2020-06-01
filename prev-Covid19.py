import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import metrics
from sklearn.model_selection import train_test_split

#get data

data = pd.read_csv("covid_final.csv")

data['indice de Gini'] = data['indice de Gini'].str.replace('[A-Za-z]', '').str.replace('.', '').str.replace(',', '.').astype(float)
data['IDHM 2010'] = data['IDHM 2010'].str.replace('[A-Za-z]', '').str.replace('.', '').str.replace(',', '.').astype(float)
data['IDHM Renda 2010'] = data['IDHM Renda 2010'].str.replace('[A-Za-z]', '').str.replace('.', '').str.replace(',', '.').astype(float)
data['IDHM Longevidade 2010'] = data['IDHM Longevidade 2010'].str.replace('[A-Za-z]', '').str.replace('.', '').str.replace(',', '.').astype(float)
data['IDHM Educacao 2010'] = data['IDHM Educacao 2010'].str.replace('[A-Za-z]', '').str.replace('.', '').str.replace(',', '.').astype(float)

#drop some cities
#data.drop(data[data.confirmed > 5000].index, inplace=True)

X = data.iloc[:, 3:data.shape[1]].values
y = data.iloc[:,1:2].values



#normalization

scaler_x = StandardScaler()
scaler_y = StandardScaler()

X = scaler_x.fit_transform(X)
y = scaler_y.fit_transform(y)



#split data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)

#Linear Regression

from sklearn.linear_model import LinearRegression
reglin = LinearRegression()
reglin.fit(X_train, y_train)
y_prev = reglin.predict(X_test)


result = scaler_y.inverse_transform(y_prev)
y_test = scaler_y.inverse_transform(y_test)

#result

print(metrics.mean_absolute_error(y_test, y_prev))
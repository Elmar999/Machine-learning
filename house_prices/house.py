import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 

train = pd.read_csv('C:/Users/mrelm/Desktop/kaggle/house_prices/train.csv')
test1 = pd.read_csv('C:/Users/mrelm/Desktop/kaggle/house_prices/test.csv')
test = pd.read_csv('C:/Users/mrelm/Desktop/kaggle/house_prices/test.csv')

#describe = train.info()

# print(train.columns)


# ANALYSING DATA


def scat(var):
	plt.scatter(x = train[var] , y = train['SalePrice'])
	plt.show()

def matrix_cor():
    corrmat = train.corr()
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat , vmax=.8, square=True)
    plt.show()


def correlogram(cols):
    sns.set()
    sns.pairplot(train[cols], size = 2)
    plt.show()



train_columns = list(train)


numerical_cols = []
categorical_cols = []
missing_cols = []
dataset = pd.concat([train , test],sort = False)
dataset.drop(columns= 'SalePrice' , axis = 1)
# print(dataset.columns)
# print(dataset['Street'].dtype)


train_columns.remove('SalePrice')

for i in train_columns:
    if dataset[i].dtype == 'int64' or dataset[i].dtype == 'float64' or dataset[i].dtype == 'int32':
        numerical_cols.append(i)
    else:
        categorical_cols.append(i)
	

# finding missing values in our dataframe between numerical values
#print("\tpercentage of missing values")
for i in numerical_cols:
    if dataset[i].isnull().sum() != 0:
        percent = round(dataset[i].isnull().sum()*100/len(dataset[i]),2)
        missing_cols.append(i)


# finding missing values in our dataframe between categorical values
for i in categorical_cols:
    if dataset[i].isnull().sum() != 0:
        percent = round(dataset[i].isnull().sum()*100/len(dataset[i]),2)
        if percent > 25: # if we have missing values over 25 percent we`ll drop it
            dataset.drop(columns=i , axis = 1)
        missing_cols.append(i)

# fill missing values in categorical and numerical values
for i in missing_cols:
	if dataset[i].dtype == 'object':
		dataset[i] = dataset[i].fillna(dataset[i].mode().iloc[0])	 
	else:
		dataset[i] = dataset[i].fillna(dataset[i].mean())





#   FEATURE ENGINEERING
# print(dataset['BsmtHalfBath'].corr(dataset['SalePrice']))
# print(dataset['BsmtFullBath'].corr(dataset['SalePrice']))

dataset['BsmtOvrBath'] = dataset['BsmtHalfBath'] + dataset['BsmtFullBath']
dataset['TotalFlrSf'] = dataset['1stFlrSF'] + dataset['2ndFlrSF']
dataset['TotalBath'] = dataset['HalfBath'] + dataset['FullBath']

dropped_cols = ['BsmtHalfBath' , 'BsmtFullBath' , '1stFlrSF' ,'2ndFlrSF' , 'HalfBath' , 'FullBath' , 'MSSubClass']


for i in dropped_cols:
    dataset = dataset.drop(labels = i , axis = 1)

dataset = dataset.drop(labels='GarageArea' , axis = 1)
#dataset = dataset.drop(labels = 'TotRmsAbvGrd' , axis = 1)

train_columns = list(dataset)


#print(dataset['TotalBath'].corr(dataset['BsmtOvrBath']))

#print(dataset.columns)


# exit(0)

### MODELLING   

from sklearn.preprocessing import LabelEncoder
encode=LabelEncoder()

for i in categorical_cols:
	dataset[i] = encode.fit_transform(dataset[i])
	

train = dataset.iloc[:1460].copy()
test = dataset.iloc[1460:]

train = train.drop(labels = 'GrLivArea' , axis = 1)
test = test.drop(labels = 'GrLivArea' , axis = 1)
train = train.drop(labels = 'TotalBath' , axis = 1)
test = test.drop(labels = 'TotalBath' , axis = 1)

for i in ['MSZoning' , 'Street' , 'Alley']:
	train = train.drop(labels = i , axis = 1)
	test = test.drop(labels = i , axis = 1)

#print(train.info())
#test = dataset.drop(labels= 'SalePrice' , axis =1)
test = test.drop(labels= 'SalePrice' , axis =1)
	
x = train.drop('SalePrice' , axis = 1)
y = train.SalePrice

from sklearn.model_selection import train_test_split
x_train , x_test ,y_train, y_test = train_test_split(x,y,test_size=0.2 , random_state=10) # splitting data by 20/80 division



from sklearn.linear_model import LinearRegression
rfr = LinearRegression()

from sklearn.metrics import mean_squared_error, r2_score
rfr.fit(x_train , y_train)
rfr.score(x_test , y_test)
prediction = rfr.predict(x_test)

from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(y_test, prediction))
r2_score(prediction, y_test)

prediction = rfr.predict(test)
submission = pd.DataFrame({
    "Id": test1["Id"],
    "SalePrice": prediction
})
submission.to_csv('submission.csv', index=False)












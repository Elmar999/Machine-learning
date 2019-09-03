
# Customer attrition, also known as customer churn, customer turnover, or customer defection, is the loss of clients or customers.
#predictive analytics use churn prediction models that predict customer churn by assessing their propensity of risk to churn.


# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import KFold
from sklearn.svm import SVR


df = pd.read_csv("Telecommunication_Data.csv")
# print(df.loc[:][df['churn'] == 1]) # print the tuples whose churn column is True
# print(df.loc[:][df['churn'] == 0]) # print the tuples whose churn column is False

title_mapping = {'yes':1 , 'no' :0}
# map 'yes' or 'no' indexes to 1 , 0 respectively
df['international plan'] = df['international plan'].map(title_mapping)
df['voice mail plan'] = df['voice mail plan'].map(title_mapping)

# as phone number and state columns doesnt improve our code a lot , it is better to drop them to get rid of this features
df = df.drop(labels = ["phone number"] , axis = 1)
df = df.drop(labels = ["state"] , axis = 1)
df = df.drop(labels = ["total eve charge"] , axis = 1)
df = df.drop(labels = ["total day charge"] , axis = 1)
df = df.drop(labels = ["total night charge"] , axis = 1)
df = df.drop(labels = ["total intl charge"] , axis = 1)
df = df.drop(labels = ["account length"] , axis = 1)

# x = df.iloc[:,:20] # extract whole data except last column
# y = df.iloc[:,20] # extract whole data for last column as a target

df["churn"] = df["churn"].astype(int)
y = df["churn"].values
x = df.drop(labels = ["churn"],axis = 1) # drop churn column from dataset

from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2 , random_state=10) # splitting data by 20/80 division

# print(X_train.shape , X_test.shape  , y_train.shape , y_test.shape)

# bar_chart function help to analize data with visualization
def bar_chart(feature , df):
  sns.set(font_scale=2)
  true = df[df['churn']==1][feature].value_counts()
  false = df[df['churn']==0][feature].value_counts()
  df = pd.DataFrame([true,false])
  df.index = ['True','False']
  df.plot(kind = 'bar' , figsize = (30,20),map=plt.cm.Reds , annot=True)
  plt.show()

def matrix_cor():
    correlation_matrix = df.corr()
    # # annot = True to print the values inside the square
    plt.figure(figsize=(40,30))
    sns.set(font_scale=3)
    ax = sns.heatmap(data=correlation_matrix , annot=True)
    plt.show()
# matrix_cor()

cor_matrix = df.corr()
cor_target = abs(cor_matrix['churn'])
relevant_figures = cor_target[cor_target > 0.2]
print(relevant_figures)
print(df[['total day minutes','customer service calls']].corr())


grd = GradientBoostingClassifier()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

x = scaler.fit_transform(x)
scores = []
best_svr = SVR(kernel='rbf')
cv = KFold(n_splits=10, random_state=42, shuffle=False) # splitting data to 10 different test and train dataset using Kfold.
for train_index, test_index in cv.split(x):
    X_train, X_test, y_train, y_test = x[train_index], x[test_index], y[train_index], y[test_index]
    grd.fit(X_train,y_train)
    scores.append(grd.score(X_test,y_test))

print(np.mean(scores))

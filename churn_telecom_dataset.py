
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

import pandas as pd

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
# df = df.drop(labels = ["international plan"] , axis = 1)
# df = df.drop(labels = ["voice mail plan"] , axis = 1)

# x = df.iloc[:,:20] # extract whole data except last column
# y = df.iloc[:,20] # extract whole data for last column as a target

df["churn"] = df["churn"].astype(int)
y = df["churn"].values
x = df.drop(labels = ["churn"],axis = 1) # drop churn column from dataset

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2 , random_state=10) # splitting data by 20/80 division


# print(X_train.shape , X_test.shape  , y_train.shape , y_test.shape)

# bar_chart function help to analize data with visualization
def bar_chart(feature , df):
  sns.set(font_scale=2)
  true = df[df['churn']==1][feature].value_counts()
  false = df[df['churn']==0][feature].value_counts()
  df = pd.DataFrame([true,false])
  df.index = ['True','False']
  df.plot(kind = 'bar' , figsize = (30,20))
  plt.show()

# bar_chart('state' , df)

# def matrix_cor():
#     correlation_matrix = df.corr()
#     # # print(correlation_matrix)
#     # # annot = True to print the values inside the square
#     plt.figure(figsize=(40,30))
#     sns.set(font_scale=3)
#     ax = sns.heatmap(data=correlation_matrix, annot=True )
#     plt.show()
# matrix_cor()
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train,y_train) # fit logisticregression modelling to our train sets
clf.predict(X_test)
score = clf.score(X_test,y_test) # i`ll show how accurate is our prediction
print(score)
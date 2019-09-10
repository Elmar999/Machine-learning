import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

sets = [train, test]
sns.set()


def drop_columns(df, label):
    df = df.drop(labels=label, axis=1)
    return df


def bar_chart(feature):
    survived = train[train['Survived'] == 1][feature].value_counts()
    dead = train[train['Survived'] == 0][feature].value_counts()
    df = pd.DataFrame([survived, dead])
    df.index = ['Survived', 'Dead']
    df.plot(kind='bar', stacked=True, figsize=(10, 5))
    plt.show()


def matrix_cor(x):
    correlation_matrix = x.corr()
    # # annot = True to print the values inside the square
    plt.figure(figsize=(10, 12))
    sns.set(font_scale=1.5)
    ax = sns.heatmap(data=correlation_matrix, annot=True, fmt=".1f" , cmap="coolwarm" , square=True)
    plt.show()


def scatter_plot(df, column1, column2):
    plt.scatter(df[column1], df[column2])
    plt.title("scatter_plot")
    plt.xlabel(column1)
    plt.ylabel(column2)
    plt.show()


# Visualization of dataset using scatter plots and bar charts
# print(train['Ticket'].describe())

# print(train.columns)
# scatter_plot(train , 'Parch','SibSp')
# bar_chart('Embarked')
# matrix_cor(train)
# print(train['Ticket'].value_counts())
# print(train.info())
# print(test.info())
# exit(0)

# DROPPING UNNECESSARY COLUMNS
# -------------------------------
train = drop_columns(train, 'Name')
test = drop_columns(test, 'Name')
train = drop_columns(train, 'Ticket')
test = drop_columns(test, 'Ticket')
train = drop_columns(train, 'Cabin')
test = drop_columns(test, 'Cabin')
train = drop_columns(train, 'SibSp')
test = drop_columns(test, 'SibSp')
# train = drop_columns(train , 'Parch')
# test = drop_columns(test , 'Parch')
train = drop_columns(train, 'PassengerId')
# test = drop_columns(test, 'PassengerId')

# matrix_cor(train)

# FILLING MISSING VALUES
# # --------------------------------
train['Age'] = train['Age'].fillna(train['Age'].mean())
test['Age'] = test['Age'].fillna(test['Age'].mean())
train['Embarked'] = train['Embarked'].fillna('S')
test['Embarked'] = test['Embarked'].fillna('S')
train['Fare'] = train['Fare'].fillna(train['Fare'].mean())
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())

# CONVERTING CATEGORICAL VALUES TO NUMERICAL
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

# train['Ticket'] = label_encoder.fit_transform(train['Ticket'])
# test['Ticket'] = label_encoder.fit_transform(test['Ticket'])
train['Sex'] = label_encoder.fit_transform(train['Sex'])
test['Sex'] = label_encoder.fit_transform(test['Sex'])
train['Embarked'] = label_encoder.fit_transform(train['Embarked'])
test['Embarked'] = label_encoder.fit_transform(test['Embarked'])

# -------------------------------------------------------


#                       TRAINING FOR OUR MODEL
# ---------------------------------------------------------


train.loc[(train.Age >= 0) & (train.Age <= 18), 'Age'] = 1
test.loc[(train.Age >= 0) & (train.Age <= 18), 'Age'] = 1
train.loc[(train.Age > 18) & (train.Age <= 35), 'Age'] = 2
test.loc[(train.Age > 18) & (train.Age <= 35), 'Age'] = 2
train.loc[(train.Age > 35) & (train.Age <= 60), 'Age'] = 3
test.loc[(train.Age > 35) & (train.Age <= 60), 'Age'] = 3
train.loc[(train.Age > 60), 'Age'] = 4
test.loc[(train.Age > 60), 'Age'] = 4
#
# train.loc[(train.Fare >= 0) & (train.Fare <= 15), 'Fare'] = 1
# test.loc[(train.Fare >= 0) & (train.Fare <= 15), 'Fare'] = 1
# train.loc[(train.Fare > 15) & (train.Fare <= 40), 'Fare'] = 2
# test.loc[(train.Fare > 15) & (train.Fare <= 40), 'Fare'] = 2
# train.loc[(train.Fare > 40) & (train.Fare <= 80), 'Fare'] = 3
# test.loc[(train.Fare > 40) & (train.Fare <= 80), 'Fare'] = 3
# train.loc[(train.Fare > 80), 'Fare'] = 4
# test.loc[(train.Fare > 80), 'Fare'] = 4


# exit(0)
#                           MODELLING
# ------------------------------------------------------
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split

# import xgboost as xgb
grd = GradientBoostingClassifier()
x = train.drop(labels='Survived', axis=1)
y = train.loc[:, 'Survived']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                    random_state=10)  # splitting data by 20/80 division
grd.fit(X_train, y_train)
score = grd.score(X_test, y_test)

# matrix_cor(train)


print(score)

test_data = test.drop("PassengerId", axis=1).copy()
prediction = grd.predict(test_data)
submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": prediction
})

submission.to_csv('submission.csv', index=False)
submission = pd.read_csv('submission.csv')
# print(submission.head())

'''import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error, mean_squared_error


df = pd.read_csv("insur.csv")
df.head()

df["sex"].replace({"male":0, "female":1}, inplace=True)
df["smoker"].replace({"yes":1, "no":1}, inplace=True)

df= pd.get_dummies(df)
#I obtain 4 new columns with the value that represent the region
df.head()

plt.figure(figsize=(10,5))

sns.heatmap(df.corr(), annot=True, cmap="Blues");
plt.show()

sns.pairplot(df);
sns.lineplot(x=df["age"], y=df["charges"]);
sns.lineplot(x=df["children"],y= df["charges"]);
sns.distplot(df["sex"], color="indigo");
sns.histplot(df["bmi"], color="grey");
sns.lineplot(x=df["bmi"], y=df["charges"], color="lime");

# Split Data in X and y
X = df.drop("charges", axis=1) # X: all columns except charges, independant variables
y = df["charges"] # y: charges colums, dependent variable
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=42) # Training dzta: 80% of the dataset; Testing Data: 20% of the dataset
len(X_train), len(X_test)

simple_model=LinearRegression()
simple_model.fit(X_train, y_train)

def model_report(y_test, y_pred):
    print(f"Accuracy: {simple_model.score(X_test, y_test) * 100:.2f}%")
    print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
    print(f"MSE: {mean_squared_error(y_test, y_pred)}")
    print(f" {r2_score(y_test, y_pred)}")

y_pred = simple_model.predict(X_test)'''

'''
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import sklearn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error, mean_squared_error


df = pd.read_csv("insur.csv")
df.head()

df["sex"].replace({"male": 0, "female": 1}, inplace=True) # Male: 0, Female: 1
df["smoker"].replace({"yes": 1, "no": 0}, inplace=True) # Yes: 1, No: 0

df = pd.get_dummies(df) # divide regions column into four differe nt column with binary labels
df.head()

plt.figure(figsize=(10,5))

sns.heatmap(df.corr(),annot=True, cmap="Blues");
plt.show()

sns.pairplot(df)
plt.show()

sns.lineplot(x=df["age"], y=df["charges"]);
sns.lineplot(x=df["children"],y= df["charges"]);
sns.distplot(df["sex"], color="indigo");
sns.histplot(df["bmi"], color="grey");
sns.lineplot(x=df["bmi"], y=df["charges"], color="lime");

# Split Data in X and y
X = df.drop("charges", axis=1) # X: all columns except charges, independant variables
y = df["charges"] # y: charges colums, dependent variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Training dzta: 80% of the dataset; Testing Data: 20% of the dataset
len(X_train), len(X_test)

simple_model=LinearRegression()
simple_model.fit(X_train, y_train)

def model_report(y_test, y_pred):
    print(f"Accuracy:{simple_model.score(X_test, y_test)*100:.2f}%")
    print(f"MAE:{mean_absolute_error(y_test, y_pred)}")
    print(f"MSE:{mean_squared_error(y_test,y_pred)}")
    print(f"R2:{r2_score(y_test,y_pred)}")
   
y_pred=simple_model.predict((X_test))

model_report(y_test, y_pred)    




model_report(y_test, y_pred)'''

'''import pandas as pd
import matplotlib.pyplot as plt

fruits = pd.read_table('fruit_data_with_colors.txt')
fruits.head()

print(fruits.shape)
print(fruits['fruit_name'].unique())
print(fruits.groupby('fruit_name').size())

import seaborn as sns
sns.countplot(x=fruits['fruit_name'],label="Count")
plt.show()

from sklearn.model_selection import train_test_split
feature_names = ['mass', 'width', 'height', 'color_score']
X = fruits[feature_names]
y = fruits['fruit_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(X_train, y_train)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'.format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'.format(knn.score(X_test, y_test)))

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
pred = knn.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))'''


import pandas as pd
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Read the xlsx file into a pandas dataframe
excel = pd.read_excel('C:\\wamp64\\www\\Fire\\forestfires3.xlsx')
# df = pd.read_excel('C:\\wamp64\\www\\Fire\\AmazonForestFires.xlsx')


excel.hist()

# Séparer les données en entrées (X) et sorties (y)
# X = df.drop("area", axis=1)
# y = df["area"]

# # Diviser les données en ensembles d'entraînement et de test
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# # Entraîner un modèle de régression logistique sur les données d'entraînement
# model = LogisticRegression()
# model.fit(X_train, y_train)

# # Prédire les résultats de test
# y_pred = model.predict(X_test)

# # Évaluer la précision du modèle
# accuracy = accuracy_score(y_test, y_pred)
# print("Précision:", accuracy)
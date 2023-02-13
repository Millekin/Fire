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


df = pd.read_excel("forestfires3.xlsx")
df.head()


df = pd.get_dummies(df) # divide regions column into four different column with binary labels
df.head()

plt.figure(figsize=(10,5))

sns.heatmap(df.corr(),annot=True, cmap="Blues");
plt.show()

sns.pairplot(df)
plt.show()


sns.lineplot(x=df["DMC"],y= df["area"], color="yellow");
sns.lineplot(x=df["FFMC"], y=df["area"], color="red");
sns.lineplot(x=df["ISI"],y= df["area"], color="blue");
sns.lineplot(x=df["temp"],y= df["area"], color="black");
sns.distplot(df["temp"], color="indigo");
#sns.histplot(df["DC"], color="purple");
sns.lineplot(x=df["DC"], y=df["area"], color="lime");


# Split Data in X and y
X = df.drop("area", axis=1) # X: all columns except charges, independant variables
y = df["area"] # y: charges colums, dependent variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Training data: 80% of the dataset; Testing Data: 20% of the dataset
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

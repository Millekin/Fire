# EE1 inter 
# VUONG Luu Anh / SALATHE Jason / BEJJANI Tala Maria / BOTAS Laura

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load the dataset from an Excel file
excel = pd.read_excel("Aff1.xlsx")

# Convert categorical variable "regions" into binary columns using one-hot encoding
excel = pd.get_dummies(excel)

# Visualize the correlation between the different variables using a heatmap
plt.figure(figsize=(10,5))
sns.heatmap(excel.corr(), annot=True, cmap="Blues")
plt.show()

# Visualize the pairwise relationship between the different variables using a pair plot
sns.pairplot(excel)
plt.show()

# Visualize the relationship between "FWI" (Forest Fire Weather Index) and "Fire" using a scatter plot
plt.scatter(excel["FWI"], excel["Fire"], color='brown')
plt.xlabel('FWI')
plt.ylabel('Fire')
plt.show()

# # Split Data in X and y
# X = excel.drop("Fire", axis=1) # X: all columns except Fire, they are the independant variables
# y = excel["Fire"] # y: area column which is the dependent variable

# X = np.array(excel['FWI']).reshape((-1, 1)) #if we only want to test one variable, here FWI for example

# We selected Temperature and FWI in X as inputs and Fire in y as the output
X=excel[['Temperature','FWI']]
y = excel['Fire']

# We split the data in 4 different sets to train, test and predict with our algorithm
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
len(X_train), len(X_test)
# We use the linear regression formula with our training data
fire_model=LinearRegression()
fire_model.fit(X_train,y_train)

# We use the predict fonction to get an output prediction : y_pred
y_pred=fire_model.predict(X_test)
# Display the different parameters of a linear regression
print(f"Mean Absolute Error (MAE): {round(mean_absolute_error(y_test,y_pred),3)}")
print(f"Mean Squared Error (MSE): {round(mean_squared_error(y_test,y_pred),3)}")
print(f"Root Mean Squared Error (RMSE) :",round(np.sqrt(mean_squared_error(y_test, y_pred)),3))
print(f"Coefficient of determination (R^2) : {round(r2_score(y_test,y_pred),3)}  /  {round((r2_score(y_test,y_pred)*100),2)}%")

# Let's round the output prediction to make it closer to the Fire output
y_pred_rounded=np.where(y_pred>0.5,1,0)
# We can display the test output and the prediction output to compare them
pred_table = pd.DataFrame({'Actual value (Test)': y_test, 'Predicted value': y_pred_rounded})

# print(pred_table)
# We can have the percentage of the same values between our 2 arrays
print("\nPercentage of accurate value between prediction and actual values :",round((np.sum(y_test == y_pred_rounded)/len(y_test))*100,2),"%")

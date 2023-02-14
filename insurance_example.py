import pandas as pd
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Read the xlsx file into a pandas dataframe
excel = pd.read_excel('C:\\wamp64\\www\\Fire\\Aff1.xlsx')

# # Split Data in X and y
# X = excel.drop("Fire", axis=1) # X: all columns except area, they are the independant variables
# y = excel["Fire"] # y: area column which is the dependent variable

X = np.array(excel['FWI']).reshape((-1, 1))
# X=excel[['Temperature','FWI']]
y = excel['Fire']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
len(X_train), len(X_test)

fire_model=LinearRegression()
fire_model.fit(X_train,y_train)

y_pred=fire_model.predict(X_test)

print(f"Mean Absolute Error (MAE): {round(mean_absolute_error(y_test,y_pred),3)}")
print(f"Mean Squared Error (MSE): {round(mean_squared_error(y_test,y_pred),3)}")
print(f"Root Mean Squared Error (RMSE) :",round(np.sqrt(mean_squared_error(y_test, y_pred)),3))
print(f"Coefficient of determination (R^2) : {round(r2_score(y_test,y_pred),3)}  /  {round((r2_score(y_test,y_pred)*100),2)}%")

y_pred_rounded=np.where(y_pred>0.5,1,0)

pred_table = pd.DataFrame({'Actual value (Test)': y_test, 'Predicted value': y_pred_rounded})

# print(pred_table)
print("\nPercentage of accurate value between prediction and actual values :",round((np.sum(y_test == y_pred_rounded)/len(y_test))*100,2),"%")

# x_new=np.array([1,30,10]).reshape(-1,1)
# y_new = fire_model.predict(x_new)
# print("Result ",y_new)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

data = pd.read_csv("SVM_workbook .csv")
data.head(5)
data.info()
data.describe()
print(data.describe())

data = data.drop(columns = ["Village"], axis = 1)

print(data.head(5))
numerical_cols = ["Temp(oC)",	 "Rain (mm)",	"Elevation(m)",	"Slope(deg)",	"AOC",	"Value of livestock",	"income alternatives",
                   "any member working as labourer",	"FM"]

print(data["Temp(oC)"].value_counts())
data = data.rename(columns = {"Farming mode": "target"})
print(data.head(2))
print(numerical_cols)

# split the dataset to training and test dataset
data_train, data_test  = train_test_split(data, test_size=0.2, random_state = 42)
print(len(data_train), len(data_test))

scaler = StandardScaler()

def get_features_and_target_arrays(data,numerical_cols, scaler):
    x_numeric_scaled = scaler.fit_transform(data[numerical_cols])
    x = x_numeric_scaled
    y = data["target"]
    
    return x,y

x_train,y_train = get_features_and_target_arrays(data_train, numerical_cols,scaler)

print(x_train)

# Train
clf = LogisticRegression()
clf = clf.fit(x_train,y_train)
x_test,y_test = get_features_and_target_arrays(data_test, numerical_cols,scaler)
test_pred = clf.predict(x_test)
test_pred_LRC = clf.decision_function(x_test)
print(mean_squared_error(y_test, test_pred))
print(accuracy_score(y_test, test_pred))

print(confusion_matrix(y_test, test_pred))
# print(clf.coef_)
print(roc_auc_score(y_test,test_pred))

clf1 = SVC()
clf1.fit(x_train,y_train)
x_test1,y_test1 = get_features_and_target_arrays(data_test, numerical_cols,scaler)
test_pred1 = clf1.predict(x_test1)
test_pred_SVC = clf1.decision_function(x_test)
print(mean_squared_error(y_test, test_pred1))
print(accuracy_score(y_test, test_pred1))
print(confusion_matrix(y_test, test_pred1))
print(roc_auc_score(y_test,test_pred1))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

data=pd.read_csv("cibil_score_dataset_final.csv")
print("no. of Null values")
print(data.isnull().sum())
columns_to_drop = ['Name','Occupation','Bank']
data = data.drop(columns=columns_to_drop, axis = 1)
X=data.drop(['Score_Category'],axis=1)
y=data['Score_Category']

# Encode the target variable 'y' before splitting
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LogisticRegression(max_iter=100000)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)
cm=confusion_matrix(y_test,y_pred)
cr=classification_report(y_test,y_pred)
print(f"accuracy:{accuracy}")
print("confusion matrix")
print(cm)
print("classification Report")
print(cr)

print("mean absolute error")
print(metrics.mean_absolute_error(y_test,y_pred))
print("mean squared error")
print(metrics.mean_squared_error(y_test,y_pred))
print("root mean squared error")
print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

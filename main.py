import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
dataset=pd.read_csv("Social_Network_Ads.csv")

X=dataset[['Age', 'EstimatedSalary']]

y=dataset['Purchased']
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.30,random_state=42)

from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=5)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

error_rate=[]
for i in range(1,50):
    model=KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    error_rate.append(np.mean(y_test != y_pred))
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test, y_pred)*100
print(accuracy,file=open("accuracy.txt", "a"))
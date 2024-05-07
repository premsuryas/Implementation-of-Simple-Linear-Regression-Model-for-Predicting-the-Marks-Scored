# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:S.PREM KUMAR 
RegisterNumber:23013598  
*/
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error , mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()

df.tail()

X = df.iloc[:,:-1].values
X

y = df.iloc[:,-1].values
y

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)

y_pred

y_test

plt.scatter(X_train,y_train,color="green")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,y_test,color="grey")
plt.plot(X_test,regressor.predict(X_test),color="purple")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

```

## Output:
![simple linear regression model for predicting the marks scored](![ml 01](https://github.com/premsuryas/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147473858/24aef182-0c1c-4599-856c-ce43f5be9678)
)
![simple linear regression model for predicting the marks scored](![ml 02](https://github.com/premsuryas/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147473858/52a93c4c-1776-47b1-be2c-96fb0e137dbc)
)
![simple linear regression model for predicting the marks scored](![ml 03](https://github.com/premsuryas/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147473858/15c157f3-7e6f-4340-8f89-a221a03651a4)
)
![simple linear regression model for predicting the marks scored](![ml 04](https://github.com/premsuryas/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147473858/87dc1be7-d3a5-40db-aeaa-8b77e9e113ff)
![simple linear regression model for predicting the marks scored](![ml 05](https://github.com/premsuryas/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147473858/4e7e42df-bba5-4bb2-b0c6-2665af038132)
)
![simple linear regression model for predicting the marks scored](![ml 06](https://github.com/premsuryas/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147473858/56079bdc-70b2-492a-bbcf-b92a15a8d57b)
)
![simple linear regression model for predicting the marks scored](![ai 06](https://github.com/premsuryas/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147473858/b0463fe7-1f0f-4780-8379-cf6093be6377)
)
![simple linear regression model for predicting the marks scored](![ai 07](https://github.com/premsuryas/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147473858/05c088bc-5fbe-49b2-b9a3-8e6190799ed6)
)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

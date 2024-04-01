# Developed by: Naveen Kumar.T
# RegisterNumber: 212223220067  
# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages.
2.Read the given dataset and assign x and y array.
3.Split x and y into training and test set.
4.Scale the x variables.
5.Fit the logistic regression for the training set to predict y.
6.Create the confusion matrix and find the accuracy score, recall sensitivity and specificity.
7.Plot the training set results.

## Program:
Program to implement the the Logistic Regression Using Gradient Descent.
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: SAKTHI AISHWARYA.S
RegisterNumber:  212219040132
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#reading and displaying dataframe
df=pd.read_csv("Social_Network_Ads (1).csv")
df
x=df.iloc[:,[2,3]].values
y=df.iloc[:,4].values 
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=0)
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
xtrain=sc_x.fit_transform(xtrain)
xtest=sc_x.fit_transform(xtest)
from sklearn.linear_model import LogisticRegression
c=LogisticRegression(random_state=0)
c.fit(xtrain,ytrain)
ypred=c.predict(xtest)
ypred
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(ytest,ypred)
cm
from sklearn import metrics
acc=metrics.accuracy_score(ytest,ypred)
acc
r_sens=metrics.recall_score(ytest,ypred,pos_label=1)
r_spec=metrics.recall_score(ytest,ypred,pos_label=0)
r_sens,r_spec
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
xs,ys=xtrain,ytrain
x1,x2=np.meshgrid(np.arange(start=xs[:,0].min()-1,stop=xs[:,0].max()+1,step=0.01),
               np.arange(start=xs[:,1].min()-1,stop=xs[:,1].max()+1,step=0.01))
plt.contourf(x1,x2,c.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
                            alpha=0.75,cmap=ListedColormap(('skyblue','green')))
plt.xlim(x1.min(),x2.max())
plt.ylim(x2.min(),x1.max())
for i,j in enumerate(np.unique(ys)):
    plt.scatter(xs[ys==j,0],xs[ys==j,1],
                c=ListedColormap(('black','white'))(i),label=j)
plt.title("Logistic Regression(Training Set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()



```

## Output:
![logistic regression using gradient descent](sam.png)


### Array of X:
Array of X:

![5 1](https://github.com/820NaveenKumar208/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/154746066/ac85f896-a9a0-40a1-8f6d-9751c93837d4)

### Array of Y:

![5 2](https://github.com/820NaveenKumar208/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/154746066/dfa2cc3d-5b40-45bb-89b5-b1367f7509df)

### Score Graph:

![5 3](https://github.com/820NaveenKumar208/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/154746066/0d8d7a8d-9639-41f8-a172-888585b35780)



### Sigmoid Function Graph:

![5 4](https://github.com/820NaveenKumar208/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/154746066/7adb40e1-b469-4de3-88a0-1bdef1364228)


### X_train_grad Value:

![5 5](https://github.com/820NaveenKumar208/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/154746066/0e7972d1-afd6-443f-ab59-4e96e12a06b6)

### Y_train_grad Value:

![5 6](https://github.com/820NaveenKumar208/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/154746066/119da7fe-6e53-4a65-afb3-bf2f5c172d3f)

### Print res_X:

![5 7](https://github.com/820NaveenKumar208/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/154746066/98b1fcfe-495b-40a5-95b1-8fed9caaf423)

### Decision boundary:

![5 8](https://github.com/820NaveenKumar208/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/154746066/3de0b481-169b-4303-adfc-5dcd65baa243)

### Probability Value:

![5 9](https://github.com/820NaveenKumar208/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/154746066/fae4dcd4-0922-45d6-8c77-82a4c5d508d6)


### Prediction Value of Mean:

![5 99](https://github.com/820NaveenKumar208/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/154746066/1679e832-178a-43cc-ace7-15ed537333f6)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.


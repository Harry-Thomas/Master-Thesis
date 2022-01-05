# -*- coding: utf-8 -*-
"""

@ author:   Harry Thomas Chirayil
This    module    contain  python specific lines    which    enables    
the    user    to    generate   logistic regression model. 


"""
# Import the required packages and libraries

import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the dataset
x = np.load('x.npy')
yold = np.load('y1_harry_test.npy')

# Clean the dataset 
floatArray = np.asarray(yold, dtype = float)
B = np.where(floatArray > 0.2, 1, 0)


scaler = StandardScaler()
x = scaler.fit_transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)

# Split the dataset into training and validation sets
X_train_full, X_test, y_train_full, y_test = train_test_split(x, B,test_size=0.2)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full,test_size=0.2)



# Instantiate the model (using the default parameters)
logreg = LogisticRegression()

print('training set: ',X_train.shape)
print('test set: ',y_train.shape)


#Reshaping as the sklearn only accepts two dimensional shape
nsamples, nx, ny = X_train.shape
d2_train_dataset = X_train.reshape((nsamples,nx*ny))

nsamples, nx, ny = X_test.shape
d2_test_dataset = X_test.reshape((nsamples,nx*ny))

# fit the model with data
logreg.fit(d2_train_dataset,y_train)

# Predict using th predict function
y_pred_logreg=logreg.predict(d2_test_dataset)

# Print the model statistics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_logreg))
print("Precision:",metrics.precision_score(y_test, y_pred_logreg))
print("Recall:",metrics.recall_score(y_test, y_pred_logreg))


# Setup for ROC curve generation

nsamples, nx, ny = X_test.shape
d2_test_dataset = X_test.reshape((nsamples,nx*ny))


y_pred_proba = logreg.predict_proba(d2_test_dataset)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# Setup for computing confusion matrix

from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred_logreg)
cnf_matrix
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# Create confusion matrix Heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')



# Classification report for analysis

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_logreg))










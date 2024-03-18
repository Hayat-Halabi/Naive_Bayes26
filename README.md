# Naive_Bayes26
Applying Naive Bayes Algorithm
``` python

%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


from sklearn.datasets import make_blobs

# make_blobs pararmeters: 
# n_samples: total # of points equally divided among clusters 
# centers: # of centers to generate 
# cluster_std: standard deviation of clusters 

X, y = make_blobs(100, 2, centers=2, random_state=2, cluster_std=1.5)

# X is a 2D array. X[:, 0] = all rows, column index 0. X[:, 1] = all rows, column index 1. 
# c: array-like or list of colors or color - 

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu');

#Observations: Scatter plot of X and y. You can see above that there are two clusters in the dataset.

from sklearn.naive_bayes import GaussianNB

# Instantiate the Gaussian NB model 
model = GaussianNB()

# Train the model 
model.fit(X, y);

# Generate new data 
# rand()- creates an array of the given shape and populates it with random values 
rng = np.random.RandomState(0)
# array shape: 2000 rows, 2 columns 
Xnew = [-6, -14] + [14, 18] * rng.rand(2000, 2) 

# Predict label of input data 
ynew = model.predict(Xnew)

# Actual values 
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
lim = plt.axis() 

# Predicted values 
plt.scatter(Xnew[:, 0], Xnew[:, 1], c=ynew, s=20, cmap='RdBu', alpha=0.1)
plt.axis(lim);

#Observations: Scatter plot of X and Y with a new prediction
#As you can see in the plot above, 2000 new predictions have been made.
#Predictions are in a light color, and the actuals are in a dark color.

# predict_proba() - predicts class probabilities 
yprob = model.predict_proba(Xnew)
yprob[-8:].round(2)

# example below: The probability for class 0 is 0.89, the probability for class 1 is 0.11 


#Obeservations

#For the first one, there is a probability of 0.89.
#The probabilities for the remaining 6 are 1 and 0.
#The probabilities for the remaining one is 0.15.






```

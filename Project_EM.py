#!/usr/bin/env python
# coding: utf-8

# # Project
# The objective of this project is to implement the Expectation-Maximization Algorithm on the Sleep in Mammals data set.

# #### Imported libraries

# In[15]:


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from functools import reduce
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut


# #### This code loads the file and the X matrix and y vector

# In[17]:


data= pd.read_csv('data\sleep2.csv')
result = pd.read_excel('data/ibmresults.xlsx')
attributes = list(data.columns)
attributes = attributes[1:10]
results = result.loc[:,'body_weight':'sleep_exposure_index']
results=results.values
X= data.loc[:,'body_weight':'sleep_exposure_index']
X= X.values
y=data.loc[:,"overall_danger_index"]
y=y.values
data.head()


# ### To do:
# + implement a different calculation for EM that wouldn't allow minus values because it doesn't fit our model 
# + it works now without gaussian mixture model so do we need the gaussian or not, if yes how do we implement it
# + figure out what the M and O is in the code below, if we want to implement different calculation we still need to know it because it seems to select the correct cells with missing values and is needed for updating them
# + implement a different way to check the accuracy of the results, because this one doesn't seem to be a reliable one $\rightarrow$ ask TA

# In[11]:


def em (X, max_iter, eps = 1e-5):
    
    nr, nc = X.shape #nr - number of rows, nc - number of columns
    C = np.isnan(X)== False #copy of X array where each missing value is stored as False and each numeric value is stored as True
    one_to_nc = np.arange(1, nc + 1, step = 1) #array of numbers of columns so [1,2,3,4,5,6,7,8,9]
    M = one_to_nc * (C == False) - 1 #array with -1 and the index value in a place of missing value
    O = one_to_nc * C - 1 #array with indexes for the values and -1 for missing values
  
    # Generate Mean and Sigma
    Mean = np.nanmean(X, axis = 0) #compute the arithmetic mean along the specified axis, ignoring NaNs
    observed_rows = np.where(np.isnan(sum(X.T)) == False)[0]  #[ 0  1  2  3 ... 61] indexes of rows where there are no missing values
    Sigma = np.cov(X[observed_rows, ].T,y[observed_rows]) #estimate a covariance matrix, given data and weights
    
    if np.isnan(Sigma).any():
        Sigma = np.diag(np.nanvar(X, axis = 0))
    
    Mean_up, Sigma_up = {}, {} #generate lists
    X_copy = X.copy() #make a copy of X
    noconv = True 
    iteration = 0
    
    #Updating
    while noconv and iteration < max_iter:
        for i in range(nr): #in range 62
            Sigma_up[i] = np.zeros(nc ** 2).reshape(nc, nc)#gives a 9x9 array of zeros for each new i
            if set(O[i, ]) != set(one_to_nc - 1): # missing component exist
                M_i, O_i = M[i, ][M[i, ] != -1], O[i, ][O[i, ] != -1] 
                #M - missing values indexes
                #O - the indexes of not missing values
                S_MM = Sigma[np.ix_(M_i, M_i)] # cross product of Sigma with the missing values indexes
                S_MO = Sigma[np.ix_(M_i, O_i)] # cross product of Sigma with the missing values and observed values indexes
                S_OM = S_MO.T #transpose 
                S_OO = Sigma[np.ix_(O_i, O_i)]  # cross product of Sigma with the observed values indexes
                Mean_up[i] = Mean[np.ix_(M_i)] + S_MO @ np.linalg.inv(S_OO) @ (X_copy[i, O_i]- Mean[np.ix_(O_i)])
                #the calculation of the missing values
                if ((Mean_up[i]<0).any()):
                    Mean_up[i] = Mean_up[i] +1
                #if the mean is negative add 1
                X_copy[i, M_i] = Mean_up[i] #update the missing values in X with mean
                S_MMO = S_MM - S_MO @ np.linalg.inv(S_OO) @ S_OM
                Sigma_up[i][np.ix_(M_i, M_i)] = S_MMO
        Mean_new = np.mean(X_copy, axis = 0) #new arithmetic mean
        Sigma_new = np.cov(X_copy.T, bias = 1) + reduce(np.add, Sigma_up.values())/nr #new sigma
        noconv = np.linalg.norm(Mean - Mean_new) >= eps or np.linalg.norm(Sigma - Sigma_new, ord = 2) >= eps
        #update mean and sigma for next iteration 
        Mean = Mean_new
        Sigma = Sigma_new
        iteration += 1
    
    result = { 
        'mu': Mean,
        'Sigma': Sigma,
        'X_imputed': X_copy,
    }
    
    return result


# #### Implementation on the dataset

# In[18]:


result_imputed = em(X,2000)
X_new=result_imputed['X_imputed']
print(X_new)
DF = pd.DataFrame(X_new) 
DF.to_csv("result.csv")
#print(result_imputed)


# #### Checking the accuracy of EM

# In[19]:


observed_rows = np.where(np.isnan(sum(X.T)) == False)[0]
X_train = X[observed_rows, ]
y_train = y[observed_rows]
neigh1 = KNeighborsClassifier(n_neighbors=1, metric='euclidean', metric_params=None)
neigh1.fit(X_train, y_train)
prediction1 = neigh1.predict(X_new)
prediction2 = neigh1.predict(results)
acc=accuracy_score(y,prediction1)
acc1=accuracy_score(y,prediction2)
print(y)
print(prediction1)
print("Accuracy: ",acc)
print("Accuracy: ",acc1)


# In[14]:


labels = pd.Series(y)
yz = preprocessing.LabelEncoder()
y=yz.fit_transform(labels)
loo = LeaveOneOut()
loo.get_n_splits(X_new)
observed_rows = np.where(np.isnan(sum(X.T)) == False)[0]
X_train = X[observed_rows, ]
y_train = y[observed_rows]
mean_errors=[]
for k in range(1,41):
    mean=[]
    for train_index, test_index in loo.split(X_new):
        X_test =  X_new[test_index]
        y_test =  y[test_index]
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(X_train, y_train)
        prediction = neigh.predict(X_test)
        accuracy = accuracy_score(y_test,prediction)
        error = 1-accuracy
        mean.append(error)
    mean_errors.append(np.mean(mean))
#print(mean_errors)
plot = plt.figure()
plot = plt.plot(mean_errors)
plt.xlabel("k")
plt.ylabel("Average error")
plt.title("Cross-validation average error - our algorithm")
plt.show()


# In[16]:


gmm = GaussianMixture(n_components=5).fit(X_new)
labels = gmm.predict(X_new)
plot= plt.figure(figsize=(25,25))
plot = plt.scatter(X_new[:, 0], X_new[:, 1], c=labels, s=40, cmap='viridis')
plt.show()


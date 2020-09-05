#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv    
import numpy as np
import math
import random


# In[2]:


def get_dataset():
    dataset = [] 
    with open('dataset.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            dataset.append([float(row['Year'].split('-')[0]),float(row['Days'])])
    
    return dataset

def print_stats(dataset):
    n = len(dataset)
    values = [row[1] for row in dataset]

    f_mean = 0
    for a in range(n): f_mean += values[a]
    f_mean = f_mean / n
    f_std = 0
    for b in range(n): f_std += (values[b] - f_mean)**2
    f_std = (f_std / (n - 1))**0.5
    
    print(n)
    print('{:.2f}'.format(f_mean))
    print('{:.2f}'.format(f_std))

def regression(beta_0, beta_1):
    dataset = get_dataset()
    n = len(dataset)
    features = [row[0] for row in dataset]
    values = [row[1] for row in dataset]
    mse = 0
    
    for i in range(n): mse += ((beta_0 + beta_1*features[i]) - values[i])**2
    mse = mse/n
    
    return mse

def gradient_descent(beta_0, beta_1):
    dataset = get_dataset()
    n = len(dataset)
    features = [row[0] for row in dataset]
    values = [row[1] for row in dataset]
    theta_0, theta_1 = 0, 0
    
    for i in range(n): theta_0 += (beta_0 + beta_1*features[i] - values[i])
    theta_0 = (2/n) * theta_0
    
    for i in range(n): theta_1 += (beta_0 + beta_1*features[i] - values[i])*features[i]
    theta_1 = (2/n) * theta_1
    
    return (theta_0, theta_1)
    
def iterate_gradient(T, eta):
    dataset = get_dataset()
    features = [row[0] for row in dataset]
    values = [row[1] for row in dataset]
    theta_0, theta_1 = 0, 0
    
    for i in range(1,T+1):
        pd = gradient_descent(theta_0, theta_1)
        theta_0 = theta_0 - (eta * pd[0])
        theta_1 = theta_1 - (eta * pd[1])
        print("{} {:.2f} {:.2f} {:.2f}".format(i, theta_0, theta_1, regression(theta_0, theta_1)))

def compute_betas():
    dataset = get_dataset()
    n = len(dataset)
    features = [row[0] for row in dataset]
    values = [row[1] for row in dataset]
    beta_0, beta_1 = 0, 0
    f_mean, v_mean = np.mean(features), np.mean(values)
    num, denom = 0, 0
    
    for i in range(n): num += (features[i] - f_mean) * (values[i] - v_mean)
    for i in range(n): denom += (features[i] - f_mean)**2
    beta_1 = (num/denom)
    beta_0 = v_mean - (beta_1 * f_mean)
    mse = regression(beta_0, beta_1)
    
    return (beta_0, beta_1, mse)

def predict(year):
    x = compute_betas()
    return x[0] + (x[1] * year)

def gd_helper(beta_0, beta_1, feature_dataset):
    dataset = get_dataset()
    n = len(dataset)
    features = feature_dataset
    values = [row[1] for row in dataset]
    theta_0, theta_1 = 0, 0
    
    for i in range(n): theta_0 += (beta_0 + beta_1*features[i] - values[i])
    theta_0 = (2/n) * theta_0
    
    for i in range(n): theta_1 += (beta_0 + beta_1*features[i] - values[i])*features[i]
    theta_1 = (2/n) * theta_1
    
    return (theta_0, theta_1)

def rh(beta_0, beta_1, feature_dataset):
    dataset = get_dataset()
    n = len(dataset)
    features = feature_dataset
    values = [row[1] for row in dataset]
    mse = 0
    
    for i in range(n): mse += ((beta_0 + beta_1*features[i]) - values[i])**2
    mse = mse/n
    
    return mse

def iterate_normalized(T, eta):
    dataset = get_dataset()
    n = len(dataset)
    features = [row[0] for row in dataset]
    values = [row[1] for row in dataset]
    
    f_mean = 0
    for i in range(n): f_mean += features[i]
    f_mean = f_mean/n
    
    f_std = 0
    for i in range(n): f_std += (features[i] - f_mean)**2
    f_std = (f_std/(n - 1))**0.5
    
    features = [((feature - f_mean) / f_std) for feature in features]
    beta_0, beta_1 = 0, 0
    gd_0, gd_1 = 0, 0
    mse = 0
    
    for i in range(1,T+1):
        pd = gd_helper(beta_0, beta_1, features)
        beta_0 = beta_0 - (eta * pd[0])
        beta_1 = beta_1 - (eta * pd[1])
        print("{} {:.2f} {:.2f} {:.2f}".format(i, beta_0, beta_1, rh(beta_0, beta_1,features)))

def sgd(T, eta):
    dataset = get_dataset()
    n = len(dataset)
    features = [row[0] for row in dataset]
    values = [row[1] for row in dataset]
    
    f_mean = 0
    for i in range(n): f_mean += features[i]
    f_mean = f_mean/n
    
    f_std = 0
    for i in range(n): f_std += (features[i] - f_mean)**2
    f_std = (f_std/(n - 1))**0.5
    
    features = [((feature - f_mean) / f_std) for feature in features]
    beta_0, beta_1 = 0, 0
    gd_0, gd_1 = 0, 0
    mse = 0
    
    for i in range(1,T+1):
        xj = random.choice(features)
        yj = values[features.index(xj)]
        gd_0 = 2*(beta_0 + (beta_1 * xj) - yj)
        gd_1 = 2*(beta_0 + (beta_1 * xj) - yj) * xj
        beta_0 = beta_0 - (eta * gd_0)
        beta_1 = beta_1 - (eta * gd_1)
        print("{} {:.2f} {:.2f} {:.2f}".format(i, beta_0, beta_1, rh(beta_0, beta_1,features)))
        


# In[20]:





# In[ ]:





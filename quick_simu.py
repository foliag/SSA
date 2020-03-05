#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 09:19:55 2019

@author: luna
"""


import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV, Lasso, RidgeCV
from sklearn.metrics import confusion_matrix

def norm2(x):
    return np.sqrt((x**2).sum())



def SSA(X, Y, lambdaa1, lambdaa2, tolerance, init_est="lasso", beta_hat = None):
    #iteration times k
    k = 0
    n = X.shape[0]
    p = X.shape[1]
    D = np.zeros((n-2,n))
    for i in range(n-2):
        D[i,i:(i+3)] = np.array([1,-2,1])
    DTD = D.T@D
    if init_est == "lasso":
        model_lasso = LassoCV(cv=10, max_iter=10000).fit(X,Y)
        weight = abs(model_lasso.coef_)+0.0000001
    if init_est == "mar":
        weight = np.zeros(p)
        for i in range(X.shape[1]):
            weight[i]  = abs(1/X[:,i].T.dot(X[:,i])*X[:,i].T.dot(Y))

    #initialize beta, beta for k-1, beta for k-2
    if beta_hat is None:
        beta_hat = np.zeros((n,p))
        old_beta = beta_hat.copy()      #k-1
        older_beta = old_beta.copy()    #k-2
        #iterate till convergence
        #X = X*weight
        
        while True:
            k += 1
            for j in range(p):
                t = 1
                while True:
                    #cal grad_f(beta0)
                    grad_beta0 = -X[:,j]*(Y-(X*beta_hat).sum(axis=1))/n + lambdaa2 * DTD@beta_hat[:,j]
                    #cal f(beta0)
                    f_beta0 = norm2(Y-(X*beta_hat).sum(axis=1))**2/2/n + lambdaa2/2*norm2(D@beta_hat[:,j])**2
                    #cal G(beta0,t)
                    temp = beta_hat[:,j] - t*grad_beta0
                    G_beta0 = temp*(1-t*lambdaa1/weight[j]/norm2(temp))*((1-t*lambdaa1/weight[j]/norm2(temp))>0)
                    #cal f(beta_new,t)
                    t_beta_hat = beta_hat.copy()
                    t_beta_hat[:,j] = G_beta0
                    f_beta_new = norm2(Y-(X*t_beta_hat).sum(axis=1))**2/2/n + lambdaa2/2*norm2(D@G_beta0)**2
                    #break condition
                    if f_beta_new <= f_beta0 + (grad_beta0*(G_beta0-beta_hat[:,j])).sum() + norm2(G_beta0-beta_hat[:,j])**2/2/t:
                        break
                    #else update t
                    t *= 0.8
                #update parameter
                #cal v
                v = beta_hat[:,j] + (k-2)/(k+1)*(beta_hat[:,j]-older_beta[:,j])
                #cal G(v,t)
                t_beta_hat[:,j] = v
                #cal grad_v
                grad_v = -X[:,j]*(Y-(X*t_beta_hat).sum(axis=1))/n + lambdaa2 * DTD@v
                temp = v - t*grad_v
                G_v = temp*(1-t*lambdaa1/weight[j]/norm2(temp))*((1-t*lambdaa1/weight[j]/norm2(temp))>0)
                #update beta
                beta_hat[:,j] = G_v
            #max change
            delta = np.abs(beta_hat - old_beta).max()
            if k % 50 == 0:
                print('iteration: ' + str(k) + ' ========== max change: ' + str(delta))
            if delta < tolerance or k >59999:
                #beta_hat = beta_hat*weight
                break
            #else update beta_k-1 adn beta_k-2
            older_beta = old_beta.copy()
            old_beta = beta_hat.copy()

    return beta_hat


def step2_CD(X, Y, lambdaa2, tolerance2=10**-5):
    k =0
    n = X.shape[0]
    p = X.shape[1]
    beta_hat = np.zeros((n,p))
    old_beta = beta_hat.copy()
    D = np.zeros((n-2,n))
    for i in range(n-2):
        D[i,i:(i+3)] = np.array([1,-2,1])
    DTD = D.T@D
    while True:
        k += 1
        for j in range(p):
            X_j = np.delete(X,j,axis=1)
            beta_j = np.delete(beta_hat,j,axis=1)
            fac1 = lambdaa2*DTD + np.diag(X[:,j]**2)/n
            fac2 = X[:,j]*(Y-(X_j*beta_j).sum(axis=1))/n
            beta_hat[:,j] = np.linalg.inv(fac1)@fac2
        delta = np.abs(beta_hat - old_beta).max()
        if k % 50 == 0:
            print('step 2 iteration: ' + str(k) + ' ========== max change: ' + str(delta))
        if delta < tolerance2:
            break
        #else update beta_k-1
        old_beta = beta_hat.copy()
    return beta_hat




tolerance = 1e-3

import sys
sys.path.append(r'/pathway/of/gen_data.py')
import gen_data


#data generation

beta, beta_begin = gen_data.gen_beta(200,500,20,'sin')
Y,X = gen_data.simuGenerator(200,500,0.3)

#variable identification

res = SSA(X,Y,0.5,0.1,tolerance,D,DTD,init_est = "lasso")
res = SSA(X,Y,15,0.1,tolerance,D,DTD,init_est = "mar")

#re-estimation

sig_index = np.where(np.apply_along_axis(norm2, 0, res)!=0)[0]

X_sig = X[:,sig_index]
res_sig = step2_CD(X_sig, Y, 500, DTD, tolerance)

#evaluation
n, p =  X.shape
comp_beta = np.zeros((n,p))
comp_beta[:,sig_index] = res_sig
rmse = norm2(comp_beta-beta)/np.sqrt(p)
rpe = norm2(Y-(X*comp_beta).sum(axis = 1))/np.sqrt(n)
tp = sum(sig_index < sum(np.apply_along_axis(norm2, 0, res)!=0))
fp = len(sig_index) - tp


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
#import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV, Lasso, RidgeCV
from sklearn.metrics import confusion_matrix

def norm2(x):
    return np.sqrt((x**2).sum())

  

class SmoothingSubgroupAnalysis():

    def __init__(self, X, Y, K, beta, tolerance = 10**-3, lambdaPercent=500,num_lambda1=10,weight = 'lasso'):
        '''
        initiate: X: independent variables
                  Y: dependent variables
                  K: K-fold cv
                  tolerance: convergence condition
                  selection_type: model(lambda) selection evaluation type: bic/aic/cv
                  train_n: sample size used for training (only for cv)
                  test_n: sample size used for testing (only for cv)                  
        '''
        self.X = X
        self.Y = Y
        self.K = K        
        self.n = X.shape[0]
        self.p = X.shape[1]
        self.tolerance = tolerance
        self.lambdaPercent = lambdaPercent
        self.num_lambda1 = num_lambda1
        self.w = weight
        self.beta = beta
    
    def calDD(self):
        self.D = np.zeros((self.n-2,self.n))
        for i in range(self.n-2):
            self.D[i,i:(i+3)] = np.array([1,-2,1])
        self.DTD = self.D.T@self.D
    
    def Weight(self):
        if self.w == 'lasso':            
            model_lasso = LassoCV(cv=10, max_iter=10000).fit(self.X,self.Y)
            self.weight = abs(model_lasso.coef_)+0.00001
        elif self.w == 'ridge':
            model_ridge = RidgeCV(cv=10).fit(self.X,self.Y)
            self.weight = abs(model_ridge.coef_)
        elif self.w == 'ols':
            self.weight = np.zeros(self.p)
            for i in range(self.X.shape[1]):  
               self.weight[i]  = abs(1/self.X[:,i].T.dot(self.X[:,i])*self.X[:,i].T.dot(self.Y))

                    
    def SSA(self, x, y, lambdaa1, lambdaa2, beta_hat = None):
        #iteration times k
        k = 0
        n = x.shape[0]
        p = x.shape[1]
        #initialize beta, beta for k-1, beta for k-2
        if beta_hat is None:
            #star = np.ones((n,p))
            #beta_hat = model_lasso.coef_.T*star
            beta_hat = np.zeros((n,p))
        old_beta = beta_hat.copy()      #k-1
        older_beta = old_beta.copy()    #k-2
        #iterate till convergence
        while True:
            k += 1
            for j in range(self.p):
                t = 1
                while True:
                    #cal grad_f(beta0)
                    grad_beta0 = -x[:,j]*(y-(x*beta_hat).sum(axis=1))/n + lambdaa2 * self.DTD@beta_hat[:,j]
                    #cal f(beta0)
                    f_beta0 = norm2(y-(x*beta_hat).sum(axis=1))**2/2/n + lambdaa2/2*norm2(self.D@beta_hat[:,j])**2
                    #cal G(beta0,t)
                    temp = beta_hat[:,j] - t*grad_beta0
                    G_beta0 = temp*(1-t*lambdaa1/self.weight[j]/norm2(temp))*((1-t*lambdaa1/self.weight[j]/norm2(temp))>0)
                    #cal f(beta_new,t)
                    t_beta_hat = beta_hat.copy()
                    t_beta_hat[:,j] = G_beta0
                    f_beta_new = norm2(y-(x*t_beta_hat).sum(axis=1))**2/2/n + lambdaa2/2*norm2(self.D@G_beta0)**2
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
                grad_v = -x[:,j]*(y-(x*t_beta_hat).sum(axis=1))/n + lambdaa2 * self.DTD@v
                temp = v - t*grad_v
                G_v = temp*(1-t*lambdaa1/self.weight[j]/norm2(temp))*((1-t*lambdaa1/self.weight[j]/norm2(temp))>0)
                #update beta
                beta_hat[:,j] = G_v
            #max change
            delta = np.abs(beta_hat - old_beta).max()
            if k % 50 == 0:
                print('iteration: ' + str(k) + ' ========== max change: ' + str(delta))
            if delta < self.tolerance or k >69999:
                break
            #else update beta_k-1 adn beta_k-2
            older_beta = old_beta.copy()
            old_beta = beta_hat.copy()
    
        return beta_hat
    
    def step2_CD(self, X, Y, lambdaa2):
        k = 0
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
            #if k % 50 == 0:
                #print('step 2 iteration: ' + str(k) + ' ========== max change: ' + str(delta))
            if delta <  self.tolerance or k > 29999:
                break
            #else update beta_k-1
            old_beta = beta_hat.copy()
        return beta_hat
 
    def calMaxlambda1(self):
        corr = np.apply_along_axis(lambda x:x*self.Y, 0, self.X)
        #self.max_lambda1 = max(np.apply_along_axis(norm2, 0 , corr)/self.n)*max(self.weight)
        self.max_lambda1 = max(self.weight*np.apply_along_axis(norm2, 0 , corr)/self.n)
     
    def lambda1Interval(self):
        self.min_lambda1 = self.max_lambda1 / self.lambdaPercent
        #self.lambda1_interval = np.linspace(self.max_lambda1,self.min_lambda1,self.num_lambda1)
        self.lambda1_interval = np.geomspace(self.max_lambda1,self.min_lambda1,self.num_lambda1)
        #self.lambda1 = list(map(lambda x: math.log(x), np.linspace(math.e**self.max_lambda1,math.e**self.min_lambda1,self.num_lambda1)))
              
    def evaluate(self, beta_hat, q):
        beta_indic = np.concatenate((np.ones(q),np.zeros(self.p-q)))
        beta_hat_indic = np.sign(np.apply_along_axis(norm2, 0, beta_hat))
        confu = confusion_matrix(beta_indic, beta_hat_indic)
        fp_n = confu[0][1]
        fn_n = confu[1][0]
        tp_n = confu[1][1]
        precision = tp_n/(tp_n+fp_n)
        recall = tp_n/(tp_n+fn_n)
        f_score = 2*precision*recall/(precision+recall)
        #TruePositve.append(tp_n)
        #FalsePositive.append(fp_n)
        #F_score.append(f_score)
        indic_path = (np.apply_along_axis(norm2, 0, beta_hat)!=0)
        beta_mse = np.sqrt(norm2(self.beta-beta_hat)**2/self.p)
        y_mse = np.sqrt(norm2(self.Y-(self.X*beta_hat).sum(axis = 1))**2/self.n)
        #beta_loss.append(beta_mse)    
        return tp_n, fp_n, f_score, indic_path, beta_mse, y_mse
    

    def train(self):
        self.score_lambda = np.zeros((self.num_lambda1, len(lambda2)))
        self.best_score = 10000000
        self.Path = np.zeros((self.num_lambda1,(self.p+1)))
        self.TruePositive = []
        self.FalsePositive = []
        self.F_score = []
        self.Beta_mse = []
        self.pool = pd.DataFrame(columns = ['lambda1','TruePositive','FalsePositive','F_score','Beta_rmse','Y_rpe'])
        X_sig_ind_old = []
        for i,lambdaa1 in enumerate(self.lambda1_interval):
            print('=========  lambda1: round: '+ str(i+1) + ' =========')

            if i == 0:
                res = self.SSA(self.X, self.Y, lambdaa1, 0.1)
            else:
                res = self.SSA(self.X, self.Y, lambdaa1, 0.1,res)
                
            X_sig_ind = np.where(np.apply_along_axis(norm2, 0, res)!=0)[0]
            X_sig = self.X[:,X_sig_ind]
            if set(X_sig_ind)-set(X_sig_ind_old)==set():
                print('stop step 2')
                continue
            for j,lambdaa2 in enumerate(lambda2):
                print('==== current lambda2:'+ str(lambdaa2) +' lambda1:'+ str(lambdaa1) + ' ====')
                X_sig_ind_old = X_sig_ind.copy()
                res_sig = self.step2_CD(X_sig, self.Y, lambdaa2)
                score = self.cross_validation(X_sig, self.Y, lambdaa2)
                self.score_lambda[i][j]=score
                #cal confusionmatrix & F_score
                tp_n, fp_n, f_score, indic_path,_,_ = self.evaluate(res, q)
                beta_mse1 = norm2(self.beta[:,np.apply_along_axis(norm2, 0, res)!=0]-res_sig)**2
                beta_mse2 = (self.beta[:,np.apply_along_axis(norm2, 0, res)==0]**2).sum()
                beta_mse = np.sqrt((beta_mse1 + beta_mse2)/self.p)
                y_mse = np.sqrt(norm2(self.Y-(X_sig*res_sig).sum(axis = 1))**2/self.n)
                self.TruePositive.append(tp_n)
                self.FalsePositive.append(fp_n)
                self.F_score.append(f_score)
                self.Beta_mse.append(beta_mse)
                self.Path[i,0] = lambdaa1
                self.Path[i,1:] = indic_path
                poo = pd.DataFrame({'lambda2':lambdaa2,'lambda1':lambdaa1,'score': self.score_lambda[i][j],'TruePositive': tp_n,'FalsePositive':fp_n, 'F_score':f_score, 'Beta_rmse':beta_mse,'Y_rpe':y_mse},index=["0"])
                self.pool = self.pool.append(poo,ignore_index=True)
                if score < self.best_score:
                    self.best_score = score
                    self.best_lambda = {'lambda1':lambdaa1, 'lambda2':lambdaa2}
                    self.best_res = res
                    self.best_res_fina = res_sig
                    self.X_sig =X_sig
                    self.beta_mse_ic = beta_mse
#            if  tp_n == q or fp_n > 40:
#                break
        self.Path = np.delete(self.Path,np.where(self.Path.sum(axis=1)==0)[0],axis=0)
        print('best_estimate')
        #self.best_res_fina = self.step2_CD(self.X_sig, self.Y, self.best_lambda['lambda2'])
        self.tp_n_ic, self.fp_n_ic, self.f_score_ic, self.indic_path_ic,_,_ = self.evaluate(self.best_res, q)
        #self.beta_mse_ic = norm2(self.beta[:,np.apply_along_axis(norm2, 0, self.best_res)!=0]-self.best_res_fina)**2/self.n + (self.beta[:,np.apply_along_axis(norm2, 0, self.best_res)==0]**2).sum()/self.n
        self.y_mse_ic = norm2(self.Y-(self.X_sig*self.best_res_fina).sum(axis = 1))**2/self.n
        self.poo_bst_ic = pd.DataFrame({'lambda1_bst':self.best_lambda['lambda1'],'lambda2_bst':self.best_lambda['lambda2'],'score_bst': self.best_score,'TruePositive': self.tp_n_ic,'FalsePositive':self.fp_n_ic, 'F_score':self.f_score_ic, 'Beta_mse':self.beta_mse_ic,'Y_mse':self.y_mse_ic},index=["0"])

    def cross_validation(self,x,y,lambdaa2):
        test_n = int(self.n/self.K)
        cv_ind = np.arange(0,self.n,1).reshape(test_n,self.K)
        cv_c = cv_ind.shape[1]
        p = x.shape[1]
        self.score_lambda = np.zeros((self.num_lambda1, len(lambda2)))
        y_mse = []
        for l in range(cv_c):
            print('+++++++ cv_group: '+ str(l) + ' +++++++++')
            test_x = x[cv_ind[:,l]]
            test_y = y[cv_ind[:,l]]
            train_x = np.delete(x,cv_ind[:,l],axis = 0)
            train_y = np.delete(y,cv_ind[:,l],axis = 0)
            res = self.step2_CD(train_x, train_y, lambdaa2)
            #cal beta_test_hat
            beta_test_hat = np.zeros(test_n*p).reshape(test_n,p)
            for m in range(test_n):
                if cv_ind[m,l] == 0:
                    beta_test_hat[m,:] = res[0,:]
                elif cv_ind[m,l] == self.n-1:
                    beta_test_hat[m,:] = res[-1,:]
                else:
                    train_neibor_ori = cv_ind[m,l]-1
                    train_neibor = train_neibor_ori - (train_neibor_ori//self.K+train_neibor_ori//(l+1+self.K*m))
                    beta_test_hat[m,:] = res[train_neibor,:]
            y_mse.append(norm2(test_y-(test_x*beta_test_hat).sum(axis = 1))**2/test_n)
        mean_score = np.array(y_mse).mean()
        return mean_score    
    
    def compare_lasso(self):
        model_lasso = LassoCV(cv=10, max_iter=10000).fit(self.X,self.Y)
        self.lambda_lasso = model_lasso.alphas_ 
        self.coef_lasso = model_lasso.coef_
        self.lasso_best_lambda = model_lasso.alpha_
        self.lasso_best_beta = model_lasso.coef_*np.ones((self.n,self.p))
        self.Path_lasso = np.zeros((len(self.lambda_lasso),(self.p+1)))
        self.pool_lasso = pd.DataFrame(columns = ['lambda_lasso','TruePositve_lasso','FalsePositive_lasso','F_score_lasso','Beta_rmse_lasso','Y_rpe_lasso'])
        ###iteration for each lambda_lasso
        for i, la_lambda in enumerate(self.lambda_lasso):
            model_lasso_iter = Lasso(la_lambda).fit(self.X,self.Y)
            model_lasso_iter_beta = model_lasso_iter.coef_*np.ones((self.n,self.p))
            tp_n_lasso, fp_n_lasso, f_score_lasso, indic_path_lasso, beta_mse_lasso, y_mse_lasso = self.evaluate(model_lasso_iter_beta, q)
            self.Path_lasso[i,0] = la_lambda
            self.Path_lasso[i,1:(self.p+1)] = indic_path_lasso
            la_poo = pd.DataFrame({'lambda_lasso':la_lambda,'TruePositve_lasso': tp_n_lasso,'FalsePositive_lasso':fp_n_lasso, 'F_score_lasso':f_score_lasso, 'Beta_rmse_lasso':beta_mse_lasso,'Y_rpe_lasso':y_mse_lasso},index=["0"])
            self.pool_lasso = self.pool_lasso.append(la_poo,ignore_index=True)




q = 20
lambda2 = [10, 200, 500]


import sys
sys.path.append(r'/pathway/of/gen_data.py')
import gen_data

#data generation

beta, beta_begin = gen_data.gen_beta(100,50,20,'sin')
Y,X = gen_data.simuGenerator(100,50,0.3,beta)

##SSA
fina2 = SmoothingSubgroupAnalysis(X, Y, 5,  beta, weight = 'lasso',lambdaPercent=500,num_lambda1=3)
fina2.calDD()
fina2.Weight()
fina2.calMaxlambda1()
fina2.lambda1Interval()
fina2.train()    #where algorithm starts


evaluate_result = fina2.pool    #evaluation results for all candidate tuning parameters
best_beta_est = fina2.best_res_fina   #coefficient estimation results under the best parameter sets
path = fina2.Path     #entering pathway of significant coefficients for each lambda1
best_result = fina2.poo_bst_ic    # evaluation results under the best parameter sets


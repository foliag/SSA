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



def gen_beta(n,p,q,beta_type = None):
    ####generate beta
    '''
    first q columns of beta has influence
    they begin with U(0,1)
    each individual plus 0.1
    then randomly use their sin/exp/ln/quadratic
    '''
    
    
    #np.random.seed(300)

    #initialize true beta
    beta = np.zeros((n,p))
    type_list = ['sin', 'exp', 'ln', 'linear', 'mix1', 'mix2', 'mix3']
    for i in range(q):
        if beta_type == None:
            t = type_list[np.random.randint(0, len(type_list))]
        else:
            t = beta_type
            
        print('beta_' + str(i) +' is generated from ' + t)
        k = q//4
       
        if t == 'sin':
            beta_begin = np.random.uniform(0,0.5,q)
            for i in range(0,k):
                u = beta_begin[i] + np.linspace(0,n-1,n)/10
                beta[:,i] = 0.02*n*np.sin(20/n*np.pi*u)+5
            for i in range(k,2*k):
                u = beta_begin[i] + np.linspace(0,n-1,n)/10
                beta[:,i] = 0.02*n*np.cos(16/n*np.pi*u+0.5)+5
            for i in range(2*k,3*k):
                u = beta_begin[i] + np.linspace(0,n-1,n)/10
                beta[:,i] = 0.02*n*np.sin(15/n*np.pi*u-0.8)+5
            for i in range(3*k,q):
                u = beta_begin[i] + np.linspace(0,n-1,n)/10
                beta[:,i] = 0.02*n*np.sin(20/n*np.pi*u)+5                
       
        if t == 'exp':
            beta_begin = np.random.uniform(0,0.4,q)
            for i in range(k):
                u = beta_begin[i] + np.linspace(0,n-1,n)/100
                beta_exp3 = np.array(list(map(lambda x: math.exp(x), -1*u)))
                beta[:,i] = 10*beta_exp3+5
            for i in range(k,2*k):
                u = beta_begin[i] + np.linspace(0,n-1,n)/100
                beta_exp3 = np.array(list(map(lambda x: math.exp(x), -1.3*u)))
                beta[:,i] = 10*beta_exp3+5
            for i in range(2*k,3*k):
                u = beta_begin[i] + np.linspace(0,n-1,n)/100
                beta_exp3 = np.array(list(map(lambda x: math.exp(x), -1.8*u)))
                beta[:,i] = 10*beta_exp3+5
            for i in range(3*k,q):
                u = beta_begin[i] + np.linspace(0,n-1,n)/100
                beta_exp3 = np.array(list(map(lambda x: math.exp(x), -2*u)))
                beta[:,i] = 10*beta_exp3+5

        if t == 'ln':
            beta_begin = np.random.uniform(0.3,0.7,q)
            for i in range(k):
                u = beta_begin[i] + np.linspace(0,n-1,n)/10
                beta_ln = np.array(list(map(lambda x: math.log(x), u**4)))
                beta[:,i] = 0.5*beta_ln+5
            for i in range(k,2*k):
                u = beta_begin[i] + np.linspace(0,n-1,n)/10
                beta_ln = np.array(list(map(lambda x: math.log(x), u**4)))
                beta[:,i] = 0.5*beta_ln+7
            for i in range(2*k,3*k):
                u = beta_begin[i] + np.linspace(0,n-1,n)/10
                beta_ln = np.array(list(map(lambda x: math.log(x), u**3)))
                beta[:,i] = 0.5*beta_ln+5
            for i in range(3*k,q):
                u = beta_begin[i] + np.linspace(0,n-1,n)/10
                beta_ln = np.array(list(map(lambda x: math.log(x), u**3)))
                beta[:,i] = 0.5*beta_ln+7
            
        if t == 'linear':
            beta_begin = np.random.uniform(0,1,q)
            for i in range(q):
                u = beta_begin[i] + np.linspace(0,n-1,n)/10
                beta[:,i] = 0.5*u+5
            for i in range(k,2*k):
                u = beta_begin[i] + np.linspace(0,n-1,n)/10
                beta[:,i] = 0.8*u+5
            for i in range(2*k,3*k):
                u = beta_begin[i] + np.linspace(0,n-1,n)/10
                beta[:,i] = 0.5*u+3   
            for i in range(3*k,q):
                u = beta_begin[i] + np.linspace(0,n-1,n)/10
                beta[:,i] = 0.8*u+3
        
        if t == 'mix1':
            u = np.zeros((k,n))
            beta_begin = np.random.uniform(0,0.5,k)
            for i in range(k):
                u[i,:] = beta_begin[i] + np.linspace(0,n-1,n)/10
            beta[:,0] = 0.02*n*np.sin(20/n*np.pi*u[0,:])+5
            beta[:,1] = 0.02*n*np.sin(15/n*np.pi*u[1,:]-0.8)+5
            beta[:,2] = 0.02*n*np.cos(16/n*np.pi*u[2,:]+0.5)+5
            beta[:,3] = 0.02*n*np.sin(15/n*np.pi*u[3,:]-0.8)+5
            beta[:,4] = 0.02*n*np.sin(20/n*np.pi*u[4,:])+5
            if q == 40:
                beta_begin = np.random.uniform(0,0.5,k)
                for i in range(k):
                    u[i,:] = beta_begin[i] + np.linspace(0,n-1,n)/10
                beta[:,20] = 0.02*n*np.sin(20/n*np.pi*u[0,:])+5
                beta[:,21] = 0.02*n*np.sin(15/n*np.pi*u[1,:]-0.8)+5
                beta[:,22] = 0.02*n*np.cos(16/n*np.pi*u[2,:]+0.5)+5
                beta[:,23] = 0.02*n*np.sin(15/n*np.pi*u[3,:]-0.8)+5
                beta[:,24] = 0.02*n*np.sin(20/n*np.pi*u[4,:])+5

            beta_begin = np.random.uniform(0,0.4,k)
            for i in range(k):
                u[i,:] = beta_begin[i] + np.linspace(0,n-1,n)/100
                
            beta_exp3 = np.array(list(map(lambda x: math.exp(x), -1.3*u[0,:])))
            beta[:,5] = 10*beta_exp3+5
            beta_exp3 = np.array(list(map(lambda x: math.exp(x), -1.3*u[1,:])))
            beta[:,6] = 10*beta_exp3+5
            beta_exp3 = np.array(list(map(lambda x: math.exp(x), -1.3*u[2,:])))
            beta[:,7] = 10*beta_exp3+5
            beta_exp3 = np.array(list(map(lambda x: math.exp(x), -1.5*u[3,:])))
            beta[:,8] = 10*beta_exp3+5
            beta_exp3 = np.array(list(map(lambda x: math.exp(x), -1.3*u[4,:])))
            beta[:,9] = 10*beta_exp3+5
            if q == 40:
                beta_begin = np.random.uniform(0,0.4,k)
                for i in range(k):
                    u[i,:] = beta_begin[i] + np.linspace(0,n-1,n)/100
                    
                beta_exp3 = np.array(list(map(lambda x: math.exp(x), -1.3*u[0,:])))
                beta[:,25] = 10*beta_exp3+5
                beta_exp3 = np.array(list(map(lambda x: math.exp(x), -1.3*u[1,:])))
                beta[:,26] = 10*beta_exp3+5
                beta_exp3 = np.array(list(map(lambda x: math.exp(x), -1.3*u[2,:])))
                beta[:,27] = 10*beta_exp3+5
                beta_exp3 = np.array(list(map(lambda x: math.exp(x), -1.5*u[3,:])))
                beta[:,28] = 10*beta_exp3+5
                beta_exp3 = np.array(list(map(lambda x: math.exp(x), -1.3*u[4,:])))
                beta[:,29] = 10*beta_exp3+5
    
                

            beta_begin = np.random.uniform(0.4,0.8,k)
            for i in range(k):
                u[i,:] = beta_begin[i] + np.linspace(0,n-1,n)/10
            
            beta_ln = np.array(list(map(lambda x: math.log(x), u[0,:]**4)))
            beta[:,10] = 0.5*beta_ln+5
            beta_ln = np.array(list(map(lambda x: math.log(x), u[1,:]**4)))
            beta[:,11] = 0.5*beta_ln+7
            beta_ln = np.array(list(map(lambda x: math.log(x), u[2,:]**3)))
            beta[:,12] = 0.5*beta_ln+7
            beta_ln = np.array(list(map(lambda x: math.log(x), u[3,:]**3)))
            beta[:,13] = 0.5*beta_ln+5            
            beta_ln = np.array(list(map(lambda x: math.log(x), u[4,:]**3)))
            beta[:,14] = 0.5*beta_ln+5            
            if q == 40:
                for i in range(k):
                    u[i,:] = beta_begin[i] + np.linspace(0,n-1,n)/10
                
                beta_ln = np.array(list(map(lambda x: math.log(x), u[0,:]**4)))
                beta[:,30] = 0.5*beta_ln+5
                beta_ln = np.array(list(map(lambda x: math.log(x), u[1,:]**4)))
                beta[:,31] = 0.5*beta_ln+7
                beta_ln = np.array(list(map(lambda x: math.log(x), u[2,:]**3)))
                beta[:,32] = 0.5*beta_ln+7
                beta_ln = np.array(list(map(lambda x: math.log(x), u[3,:]**3)))
                beta[:,33] = 0.5*beta_ln+5            
                beta_ln = np.array(list(map(lambda x: math.log(x), u[4,:]**3)))
                beta[:,34] = 0.5*beta_ln+5            
    
                
            
            beta_begin = np.random.uniform(0,1,k)
            for i in range(k):
                u[i,:] = beta_begin[i] + np.linspace(0,n-1,n)/10
            
            beta[:,15] = 0.5*u[0,:]+5
            beta[:,16] = 0.8*u[1,:]+5
            beta[:,17] = 0.5*u[2,:]+3
            beta[:,18] = 0.8*u[3,:]+3
            beta[:,19] = 0.5*u[4,:]+5
            if q ==40:
                beta_begin = np.random.uniform(0,1,k)
                for i in range(k):
                    u[i,:] = beta_begin[i] + np.linspace(0,n-1,n)/10
                
                beta[:,35] = 0.5*u[0,:]+5
                beta[:,36] = 0.8*u[1,:]+5
                beta[:,37] = 0.5*u[2,:]+3
                beta[:,38] = 0.8*u[3,:]+3
                beta[:,39] = 0.5*u[4,:]+5


        if t == 'mix2':
            u = np.zeros((q,n))
            
            beta_begin1 = np.random.uniform(0,0.5,int(q/20*8))
            beta_begin2 = np.random.uniform(0,0.4,int(q/20*2))
            beta_begin3 = np.random.uniform(0.4,0.8,int(q/20*3))
            beta_begin4 = np.random.uniform(0,1,int(q/20*7))
            beta_begin = np.concatenate((beta_begin1,beta_begin2,beta_begin3,beta_begin4))

            for i in range(8):
                u[i,:] = beta_begin[i] + np.linspace(0,n-1,n)/10
            beta[:,0] = 0.02*n*np.sin(20/n*np.pi*u[0,:])+5
            beta[:,1] = 0.02*n*np.cos(16/n*np.pi*u[1,:]+0.5)+5
            beta[:,2] = 0.02*n*np.sin(15/n*np.pi*u[2,:]-0.8)+5
            beta[:,3] = 0.02*n*np.cos(16/n*np.pi*u[3,:]+0.5)+5
            beta[:,4] = 0.02*n*np.sin(20/n*np.pi*u[4,:])+5
            beta[:,5] = 0.02*n*np.sin(15/n*np.pi*u[5,:]-0.8)+5
            beta[:,6] = 0.02*n*np.cos(16/n*np.pi*u[6,:]+0.5)+5
            beta[:,7] = 0.02*n*np.sin(15/n*np.pi*u[7,:]-0.8)+5
            if q == 40:
                for i in range(20,28):
                    u[i,:] = beta_begin[i] + np.linspace(0,n-1,n)/10
                beta[:,20] = 0.02*n*np.sin(20/n*np.pi*u[20,:])+5
                beta[:,21] = 0.02*n*np.cos(16/n*np.pi*u[21,:]+0.5)+5
                beta[:,22] = 0.02*n*np.sin(15/n*np.pi*u[22,:]-0.8)+5
                beta[:,23] = 0.02*n*np.cos(16/n*np.pi*u[23,:]+0.5)+5
                beta[:,24] = 0.02*n*np.sin(20/n*np.pi*u[24,:])+5
                beta[:,25] = 0.02*n*np.sin(15/n*np.pi*u[25,:]-0.8)+5
                beta[:,26] = 0.02*n*np.cos(16/n*np.pi*u[26,:]+0.5)+5
                beta[:,27] = 0.02*n*np.sin(15/n*np.pi*u[27,:]-0.8)+5

            for i in range(8,10):
                u[i,:] = beta_begin[i] + np.linspace(0,n-1,n)/100
            beta_exp3 = np.array(list(map(lambda x: math.exp(x), -1.5*u[8,:])))
            beta[:,8] = 3*beta_exp3+5
            beta_exp3 = np.array(list(map(lambda x: math.exp(x), -u[9,:])))
            beta[:,9] = 3*beta_exp3+5
            if q == 40:
                for i in range(28,30):
                    u[i,:] = beta_begin[i] + np.linspace(0,n-1,n)/100
                beta_exp3 = np.array(list(map(lambda x: math.exp(x), -1.5*u[28,:])))
                beta[:,28] = 10*beta_exp3+5
                beta_exp3 = np.array(list(map(lambda x: math.exp(x), -u[29,:])))
                beta[:,29] = 10*beta_exp3+5

                
            for i in range(10,13):
                u[i,:] = beta_begin[i] + np.linspace(0,n-1,n)/10
            beta_ln = np.array(list(map(lambda x: math.log(x), u[10,:]**2)))
            beta[:,10] = 0.5*beta_ln+5
            beta_ln = np.array(list(map(lambda x: math.log(x), u[11,:]**2)))
            beta[:,11] = 0.5*beta_ln+5
            beta_ln = np.array(list(map(lambda x: math.log(x), u[12,:]**2)))
            beta[:,12] = 0.5*beta_ln+5
            if q == 40:
                for i in range(30,33):
                    u[i,:] = beta_begin[i] + np.linspace(0,n-1,n)/10
                beta_ln = np.array(list(map(lambda x: math.log(x), u[30,:]**2)))
                beta[:,30] = 0.5*beta_ln+5
                beta_ln = np.array(list(map(lambda x: math.log(x), u[31,:]**2)))
                beta[:,31] = 0.5*beta_ln+5
                beta_ln = np.array(list(map(lambda x: math.log(x), u[32,:]**2)))
                beta[:,32] = 0.5*beta_ln+5


            for i in range(13,20):
                u[i,:] = beta_begin[i] + np.linspace(0,n-1,n)/10
            beta[:,13] = 0.8*u[13,:]+5
            beta[:,14] = 0.5*u[14,:]+3
            beta[:,15] = 0.8*u[15,:]+3
            beta[:,16] = 0.5*u[16,:]+5
            beta[:,17] = 0.5*u[17,:]+3
            beta[:,18] = 0.8*u[18,:]+3
            beta[:,19] = 0.8*u[19,:]+5
            if q == 40:
                for i in range(33,40):
                    u[i,:] = beta_begin[i] + np.linspace(0,n-1,n)/10
                beta[:,33] = 0.8*u[33,:]+5
                beta[:,34] = 0.5*u[34,:]+3
                beta[:,35] = 0.8*u[35,:]+3
                beta[:,36] = 0.5*u[36,:]+5
                beta[:,37] = 0.5*u[37,:]+3
                beta[:,38] = 0.8*u[38,:]+3
                beta[:,39] = 0.8*u[39,:]+5


            
        if t == 'mix3':
            u = np.zeros((q,n))

        
            beta_begin1 = np.random.uniform(0,0.5,int(q/20*7))
            beta_begin2 = np.random.uniform(0,0.4,int(q/20*3))
            beta_begin3 = np.random.uniform(0.4,0.8,int(q/20*4))
            beta_begin4 = np.random.uniform(0,1,int(q/20*6))
            beta_begin = np.concatenate((beta_begin1,beta_begin2,beta_begin3,beta_begin4))
            
            for i in range(7):
                u[i,:] = beta_begin[i] + np.linspace(0,n-1,n)/10
            beta[:,0] = 0.02*n*np.sin(20/n*np.pi*u[0,:])+5
            beta[:,1] = 0.02*n*np.cos(16/n*np.pi*u[1,:]+0.5)+5
            beta[:,2] = 0.02*n*np.sin(15/n*np.pi*u[2,:]-0.8)+5
            beta[:,3] = 0.02*n*np.cos(16/n*np.pi*u[3,:]+0.5)+5
            beta[:,4] = 0.02*n*np.sin(20/n*np.pi*u[4,:])+5
            beta[:,5] = 0.02*n*np.sin(15/n*np.pi*u[5,:]-0.8)+5
            beta[:,6] = 0.02*n*np.cos(16/n*np.pi*u[6,:]+0.5)+5
            if q == 40:
                for i in range(20,27):
                    u[i,:] = beta_begin[i] + np.linspace(0,n-1,n)/10
                beta[:,20] = 0.02*n*np.sin(20/n*np.pi*u[20,:])+5
                beta[:,21] = 0.02*n*np.cos(16/n*np.pi*u[21,:]+0.5)+5
                beta[:,22] = 0.02*n*np.sin(15/n*np.pi*u[22,:]-0.8)+5
                beta[:,23] = 0.02*n*np.cos(16/n*np.pi*u[23,:]+0.5)+5
                beta[:,24] = 0.02*n*np.sin(20/n*np.pi*u[24,:])+5
                beta[:,25] = 0.02*n*np.sin(15/n*np.pi*u[25,:]-0.8)+5
                beta[:,26] = 0.02*n*np.cos(16/n*np.pi*u[26,:]+0.5)+5

            for i in range(7,10):
                u[i,:] = beta_begin[i] + np.linspace(0,n-1,n)/100
            beta_exp3 = np.array(list(map(lambda x: math.exp(x), -2*u[7,:])))
            beta[:,7] = 10*beta_exp3+5
            beta_exp3 = np.array(list(map(lambda x: math.exp(x), -1.5*u[8,:])))
            beta[:,8] = 10*beta_exp3+5
            beta_exp3 = np.array(list(map(lambda x: math.exp(x), -u[9,:])))
            beta[:,9] = 10*beta_exp3+5
            if q == 40:
                for i in range(27,30):
                    u[i,:] = beta_begin[i] + np.linspace(0,n-1,n)/100
                beta_exp3 = np.array(list(map(lambda x: math.exp(x), -2*u[28,:])))
                beta[:,27] = 10*beta_exp3+5
                beta_exp3 = np.array(list(map(lambda x: math.exp(x), -1.5*u[27,:])))
                beta[:,28] = 10*beta_exp3+5
                beta_exp3 = np.array(list(map(lambda x: math.exp(x), -u[29,:])))
                beta[:,29] = 10*beta_exp3+5

            
            for i in range(10,14):
                u[i,:] = beta_begin[i] + np.linspace(0,n-1,n)/10
            beta_ln = np.array(list(map(lambda x: math.log(x), u[10,:]**4)))
            beta[:,10] = 0.5*beta_ln+5
            beta_ln = np.array(list(map(lambda x: math.log(x), u[11,:]**4)))
            beta[:,11] = 0.5*beta_ln+7
            beta_ln = np.array(list(map(lambda x: math.log(x), u[12,:]**3)))
            beta[:,12] = 0.5*beta_ln+5
            beta_ln = np.array(list(map(lambda x: math.log(x), u[13,:]**3)))
            beta[:,13] = 0.5*beta_ln+7
            if q == 40:
                for i in range(30,34):
                    u[i,:] = beta_begin[i] + np.linspace(0,n-1,n)/10
                beta_ln = np.array(list(map(lambda x: math.log(x), u[30,:]**4)))
                beta[:,30] = 0.5*beta_ln+5
                beta_ln = np.array(list(map(lambda x: math.log(x), u[31,:]**4)))
                beta[:,31] = 0.5*beta_ln+7
                beta_ln = np.array(list(map(lambda x: math.log(x), u[32,:]**3)))
                beta[:,32] = 0.5*beta_ln+5
                beta_ln = np.array(list(map(lambda x: math.log(x), u[33,:]**3)))
                beta[:,33] = 0.5*beta_ln+7

                
            
            for i in range(14,20):
                u[i,:] = beta_begin[i] + np.linspace(0,n-1,n)/10
            beta[:,14] = 0.5*u[14,:]+5
            beta[:,15] = 0.8*u[15,:]+5
            beta[:,16] = 0.5*u[16,:]+3
            beta[:,17] = 0.5*u[17,:]+3
            beta[:,18] = 0.8*u[18,:]+3
            beta[:,19] = 0.5*u[19,:]+5
            if q == 40:
                for i in range(34,40):
                    u[i,:] = beta_begin[i] + np.linspace(0,n-1,n)/10
                beta[:,34] = 0.5*u[34,:]+5
                beta[:,35] = 0.8*u[35,:]+5
                beta[:,36] = 0.5*u[36,:]+3
                beta[:,37] = 0.5*u[37,np.rand:]+3
                beta[:,38] = 0.8*u[38,:]+3
                beta[:,39] = 0.5*u[39,:]+5

   
    return beta,beta_begin
        


       
def simuGenerator(n,p,rho):
    sigma = np.zeros(p*p).reshape(p,p)
    for i in range(p):
        for j in range(p):
            sigma[i][j]=rho**abs(i-j)
    
    X = np.random.multivariate_normal(np.array([0]*p),sigma,n)
    Y = (X*beta).sum(axis = 1) + np.random.randn(n)

    return Y, X


#beta, beta_begin = gen_beta(200,500,20,'sin')
#Y,X = simuGenerator(200,500,0.3)

#np.savetxt('beta.csv',beta,delimiter=',',fmt='%10.5f')
#np.savetxt('X.csv',X,delimiter=',',fmt='%10.5f')
#np.savetxt('Y.csv',Y,delimiter=',',fmt='%10.5f')


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
    t = beta_type
    print('beta is generated from ' + t)
    k = q//4
   
    if t == 'sin':
        beta_begin = np.random.uniform(0,0.5,q)
        for i in range(0,k):
            u = beta_begin[i] + np.linspace(0,n-1,n)/10
            beta[:,i] = 1.5*np.sin(20/n*np.pi*u)+2.5
        for i in range(k,2*k):
            u = beta_begin[i] + np.linspace(0,n-1,n)/10
            beta[:,i] = 1.5*np.cos(17/n*np.pi*u+0.4)+2.5
        for i in range(2*k,3*k):
            u = beta_begin[i] + np.linspace(0,n-1,n)/10
            beta[:,i] = 1.5*np.sin(17/n*np.pi*u-1.2)+2.5
        for i in range(3*k,q):
            u = beta_begin[i] + np.linspace(0,n-1,n)/10
            beta[:,i] = 1.5*np.cos(20/n*np.pi*u-2)+2.5        
   
    if t == 'exp':
        beta_begin = np.random.uniform(0,0.2,q)
        for i in range(k):
            u = beta_begin[i] + np.linspace(0,n-1,n)/100
            beta_exp3 = np.array(list(map(lambda x: math.exp(x), -1*u)))
            beta[:,i] = 4*beta_exp3+1
        for i in range(k,2*k):
            u = beta_begin[i] + np.linspace(0,n-1,n)/100
            beta_exp3 = np.array(list(map(lambda x: math.exp(x), -0.9*u)))
            beta[:,i] = 4*beta_exp3+1
        for i in range(2*k,3*k):
            u = beta_begin[i] + np.linspace(0,n-1,n)/100
            beta_exp3 = np.array(list(map(lambda x: math.exp(x), -0.8*u)))
            beta[:,i] = 4*beta_exp3+1
        for i in range(3*k,q):
            u = beta_begin[i] + np.linspace(0,n-1,n)/100
            beta_exp3 = np.array(list(map(lambda x: math.exp(x), -0.7*u)))
            beta[:,i] = 4*beta_exp3+1

    if t == 'ln':
        beta_begin = np.random.uniform(0.7,0.9,q)
        for i in range(k):
            u = beta_begin[i] + np.linspace(0,n-1,n)/20
            beta_ln = np.array(list(map(lambda x: math.log(x), u**3)))
            beta[:,i] = 0.5*beta_ln+1
        for i in range(k,2*k):
            u = beta_begin[i] + np.linspace(0,n-1,n)/20
            beta_ln = np.array(list(map(lambda x: math.log(x), u**2.9)))
            beta[:,i] = 0.5*beta_ln+1
        for i in range(2*k,3*k):
            u = beta_begin[i] + np.linspace(0,n-1,n)/20
            beta_ln = np.array(list(map(lambda x: math.log(x), u**2.7)))
            beta[:,i] = 0.5*beta_ln+1
        for i in range(3*k,q):
            u = beta_begin[i] + np.linspace(0,n-1,n)/20
            beta_ln = np.array(list(map(lambda x: math.log(x), u**2.5)))
            beta[:,i] = 0.5*beta_ln+1
        
    if t == 'linear':
        beta_begin = np.random.uniform(0,1,q)
        for i in range(q):
            u = beta_begin[i] + np.linspace(0,n-1,n)/10
            beta[:,i] = 0.16*u+2
        for i in range(k,2*k):
            u = beta_begin[i] + np.linspace(0,n-1,n)/10
            beta[:,i] = 0.15*u +2
        for i in range(2*k,3*k):
            u = beta_begin[i] + np.linspace(0,n-1,n)/10
            beta[:,i] = 0.14*u+2
        for i in range(3*k,q):
            u = beta_begin[i] + np.linspace(0,n-1,n)/10
            beta[:,i] = 0.13*u+2
            
    if t == 'fix':
        beta_begin = np.random.uniform(0,1,q)
        for i in range(q):
            u = beta_begin[i]*np.ones(n)
            beta[:,i] = 3*u+2
        for i in range(k,2*k):
            u = beta_begin[i]*np.ones(n)
            beta[:,i] = 2*u + 2
        for i in range(2*k,3*k):
            u = beta_begin[i]*np.ones(n)
            beta[:,i] = 2*u + 2
        for i in range(3*k,q):
            u = beta_begin[i]*np.ones(n)
            beta[:,i] = 3*u + 2

    if t == 'mix1':
        u = np.zeros((k,n))
        beta_begin = np.random.uniform(0,0.5,k)
        for i in range(k):
            u[i,:] = beta_begin[i] + np.linspace(0,n-1,n)/10
        beta[:,0] = 1.5*np.sin(20/n*np.pi*u[0,:])+2.5
        beta[:,1] = 1.5*np.cos(17/n*np.pi*u[1,:]+0.4)+2.5
        beta[:,2] = 1.5*np.sin(17/n*np.pi*u[2,:]-1.2)+2.5
        beta[:,3] = 1.5*np.cos(20/n*np.pi*u[3,:]-2)+2.5
        beta[:,4] = 1.5*np.sin(20/n*np.pi*u[4,:])+2.5
        
        if q == 40:
            beta_begin = np.random.uniform(0,0.5,k)
            for i in range(k):
                u[i,:] = beta_begin[i] + np.linspace(0,n-1,n)/10
            beta[:,20] = 1.5*np.sin(20/n*np.pi*u[0,:])+2.5
            beta[:,21] = 1.5*np.cos(17/n*np.pi*u[1,:]+0.4)+2.5
            beta[:,22] = 1.5*np.sin(17/n*np.pi*u[2,:]-1.2)+2.5
            beta[:,23] = 1.5*np.cos(20/n*np.pi*u[3,:]-2)+2.5
            beta[:,24] = 1.5*np.sin(20/n*np.pi*u[4,:])+2.5

        beta_begin = np.random.uniform(0,0.2,k)
        for i in range(k):
            u[i,:] = beta_begin[i] + np.linspace(0,n-1,n)/100
            
        beta_exp3 = np.array(list(map(lambda x: math.exp(x), -1*u[0,:])))
        beta[:,5] = 4*beta_exp3+1
        beta_exp3 = np.array(list(map(lambda x: math.exp(x), -0.9*u[1,:])))
        beta[:,6] = 4*beta_exp3+1
        beta_exp3 = np.array(list(map(lambda x: math.exp(x), -0.8*u[2,:])))
        beta[:,7] = 4*beta_exp3+1
        beta_exp3 = np.array(list(map(lambda x: math.exp(x), -0.7*u[3,:])))
        beta[:,8] = 4*beta_exp3+1
        beta_exp3 = np.array(list(map(lambda x: math.exp(x), -1*u[4,:])))
        beta[:,9] = 4*beta_exp3+1
        
        if q == 40:
            beta_begin = np.random.uniform(0,0.2,k)
            for i in range(k):
                u[i,:] = beta_begin[i] + np.linspace(0,n-1,n)/100
                
            beta_exp3 = np.array(list(map(lambda x: math.exp(x), -1*u[0,:])))
            beta[:,25] = 4*beta_exp3+1
            beta_exp3 = np.array(list(map(lambda x: math.exp(x), -0.9*u[1,:])))
            beta[:,26] = 4*beta_exp3+1
            beta_exp3 = np.array(list(map(lambda x: math.exp(x), -0.8*u[2,:])))
            beta[:,27] = 4*beta_exp3+1
            beta_exp3 = np.array(list(map(lambda x: math.exp(x), -0.7*u[3,:])))
            beta[:,28] = 4*beta_exp3+1
            beta_exp3 = np.array(list(map(lambda x: math.exp(x), -1*u[4,:])))
            beta[:,29] = 4*beta_exp3+1

            

        beta_begin = np.random.uniform(0.7,0.9,k)
        for i in range(k):
            u[i,:] = beta_begin[i] + np.linspace(0,n-1,n)/20
        
        beta_ln = np.array(list(map(lambda x: math.log(x), u[0,:]**3)))
        beta[:,10] = 0.5*beta_ln+1
        beta_ln = np.array(list(map(lambda x: math.log(x), u[1,:]**2.9)))
        beta[:,11] = 0.5*beta_ln+1
        beta_ln = np.array(list(map(lambda x: math.log(x), u[2,:]**2.7)))
        beta[:,12] = 0.5*beta_ln+1
        beta_ln = np.array(list(map(lambda x: math.log(x), u[3,:]**2.5)))
        beta[:,13] = 0.5*beta_ln+1            
        beta_ln = np.array(list(map(lambda x: math.log(x), u[4,:]**3)))
        beta[:,14] = 0.5*beta_ln+1            
        
        if q == 40:
            for i in range(k):
                u[i,:] = beta_begin[i] + np.linspace(0,n-1,n)/20
            
            beta_ln = np.array(list(map(lambda x: math.log(x), u[0,:]**3)))
            beta[:,30] = 0.5*beta_ln+1
            beta_ln = np.array(list(map(lambda x: math.log(x), u[1,:]**2.9)))
            beta[:,31] = 0.5*beta_ln+1
            beta_ln = np.array(list(map(lambda x: math.log(x), u[2,:]**2.7)))
            beta[:,32] = 0.5*beta_ln+1
            beta_ln = np.array(list(map(lambda x: math.log(x), u[3,:]**2.5)))
            beta[:,33] = 0.5*beta_ln+1            
            beta_ln = np.array(list(map(lambda x: math.log(x), u[4,:]**3)))
            beta[:,34] = 0.5*beta_ln+1            

            
        
        beta_begin = np.random.uniform(0,1,k)
        for i in range(k):
            u[i,:] = beta_begin[i] + np.linspace(0,n-1,n)/10
        
        beta[:,15] = 0.16*u[0,:]+2
        beta[:,16] = 0.15*u[1,:]+2
        beta[:,17] = 0.13*u[2,:]+2
        beta[:,18] = 0.14*u[3,:]+2
        beta[:,19] = 0.16*u[4,:]+2
        if q ==40:
            beta_begin = np.random.uniform(0,1,k)
            for i in range(k):
                u[i,:] = beta_begin[i] + np.linspace(0,n-1,n)/10
            
            beta[:,35] = 0.16*u[0,:]+2
            beta[:,36] = 0.15*u[1,:]+2
            beta[:,37] = 0.14*u[2,:]+2
            beta[:,38] = 0.13*u[3,:]+2
            beta[:,39] = 0.16*u[4,:]+2


    if t == 'mix2':
        u = np.zeros((q,n))
        
        beta_begin1 = np.random.uniform(0,0.5,int(q/20*8))
        beta_begin2 = np.random.uniform(0,0.2,int(q/20*2))
        beta_begin3 = np.random.uniform(0.7,0.9,int(q/20*3))
        beta_begin4 = np.random.uniform(0,1,int(q/20*7))
        beta_begin = np.concatenate((beta_begin1,beta_begin2,beta_begin3,beta_begin4))

        for i in range(8):
            u[i,:] = beta_begin[i] + np.linspace(0,n-1,n)/10
        beta[:,0] = 1.5*np.sin(20/n*np.pi*u[0,:])+2.5
        beta[:,1] = 1.5*np.cos(17/n*np.pi*u[1,:]+0.4)+2.5
        beta[:,2] = 1.5*np.sin(17/n*np.pi*u[2,:]-1.2)+2.5
        beta[:,3] = 1.5*np.cos(20/n*np.pi*u[3,:]-2)+2.5
        beta[:,4] = 1.5*np.sin(20/n*np.pi*u[4,:])+2.5
        beta[:,5] = 1.5*np.cos(17/n*np.pi*u[5,:]+0.4)+2.5
        beta[:,6] = 1.5*np.sin(17/n*np.pi*u[6,:]-1.2)+2.5
        beta[:,7] = 1.5*np.cos(20/n*np.pi*u[7,:]-2)+2.5
        if q == 40:
            for i in range(20,28):
                u[i,:] = beta_begin[i] + np.linspace(0,n-1,n)/10
            beta[:,20] = 1.5*np.sin(20/n*np.pi*u[20,:])+2.5
            beta[:,21] = 1.5*np.cos(17/n*np.pi*u[21,:]+0.4)+2.5
            beta[:,22] = 1.5*np.sin(17/n*np.pi*u[22,:]-1.2)+2.5
            beta[:,23] = 1.5*np.cos(20/n*np.pi*u[23,:]-2)+2.5
            beta[:,24] = 1.5*np.sin(20/n*np.pi*u[24,:])+2.5
            beta[:,25] = 1.5*np.cos(17/n*np.pi*u[25,:]+0.4)+2.5
            beta[:,26] = 1.5*np.sin(17/n*np.pi*u[26,:]-1.2)+2.5
            beta[:,27] = 1.5*np.cos(20/n*np.pi*u[27,:]-2)+2.5

        for i in range(8,10):
            u[i,:] = beta_begin[i] + np.linspace(0,n-1,n)/100
        beta_exp3 = np.array(list(map(lambda x: math.exp(x), -1*u[8,:])))
        beta[:,8] = 4*beta_exp3+1
        beta_exp3 = np.array(list(map(lambda x: math.exp(x), -0.9*u[9,:])))
        beta[:,9] = 4*beta_exp3+1
        if q == 40:
            for i in range(28,30):
                u[i,:] = beta_begin[i] + np.linspace(0,n-1,n)/100
            beta_exp3 = np.array(list(map(lambda x: math.exp(x), -1*u[28,:])))
            beta[:,28] = 4*beta_exp3+1
            beta_exp3 = np.array(list(map(lambda x: math.exp(x), -0.9*u[29,:])))
            beta[:,29] = 4*beta_exp3+1

            
        for i in range(10,13):
            u[i,:] = beta_begin[i] + np.linspace(0,n-1,n)/20
        beta_ln = np.array(list(map(lambda x: math.log(x), u[10,:]**3)))
        beta[:,10] = 0.5*beta_ln+1
        beta_ln = np.array(list(map(lambda x: math.log(x), u[11,:]**2.9)))
        beta[:,11] = 0.5*beta_ln+1
        beta_ln = np.array(list(map(lambda x: math.log(x), u[12,:]**2.7)))
        beta[:,12] = 0.5*beta_ln+1
        if q == 40:
            for i in range(30,33):
                u[i,:] = beta_begin[i] + np.linspace(0,n-1,n)/10
            beta_ln = np.array(list(map(lambda x: math.log(x), u[30,:]**3)))
            beta[:,30] = 0.5*beta_ln+1
            beta_ln = np.array(list(map(lambda x: math.log(x), u[31,:]**2.9)))
            beta[:,31] = 0.5*beta_ln+1
            beta_ln = np.array(list(map(lambda x: math.log(x), u[32,:]**2.7)))
            beta[:,32] = 0.5*beta_ln+1


        for i in range(13,20):
            u[i,:] = beta_begin[i] + np.linspace(0,n-1,n)/10
        beta[:,13] = 0.16*u[13,:]+2
        beta[:,14] = 0.15*u[14,:]+2
        beta[:,15] = 0.14*u[15,:]+2
        beta[:,16] = 0.13*u[16,:]+2
        beta[:,17] = 0.16*u[17,:]+2
        beta[:,18] = 0.15*u[18,:]+2
        beta[:,19] = 0.14*u[19,:]+2
        if q == 40:
            for i in range(33,40):
                u[i,:] = beta_begin[i] + np.linspace(0,n-1,n)/10
            beta[:,33] = 0.16*u[33,:]+2
            beta[:,34] = 0.15*u[34,:]+2
            beta[:,35] = 0.14*u[35,:]+2
            beta[:,36] = 0.13*u[36,:]+2
            beta[:,37] = 0.16*u[37,:]+2
            beta[:,38] = 0.15*u[38,:]+2
            beta[:,39] = 0.14*u[39,:]+2


        
    if t == 'mix3':
        u = np.zeros((q,n))

    
        beta_begin1 = np.random.uniform(0,0.5,int(q/20*7))
        beta_begin2 = np.random.uniform(0,0.2,int(q/20*3))
        beta_begin3 = np.random.uniform(0.7,0.9,int(q/20*4))
        beta_begin4 = np.random.uniform(0,1,int(q/20*6))
        beta_begin = np.concatenate((beta_begin1,beta_begin2,beta_begin3,beta_begin4))
        
        for i in range(7):
            u[i,:] = beta_begin[i] + np.linspace(0,n-1,n)/10
        beta[:,0] = 1.5*np.sin(20/n*np.pi*u[0,:])+2.5
        beta[:,1] = 1.5*np.cos(17/n*np.pi*u[1,:]+0.4)+2.5
        beta[:,2] = 1.5*np.sin(17/n*np.pi*u[2,:]-1.2)+2.5
        beta[:,3] = 1.5*np.cos(20/n*np.pi*u[3,:]-2)+2.5
        beta[:,4] = 1.5*np.sin(20/n*np.pi*u[4,:])+2.5
        beta[:,5] = 1.5*np.cos(17/n*np.pi*u[5,:]+0.4)+2.5
        beta[:,6] = 1.5*np.sin(17/n*np.pi*u[6,:]-1.2)+2.5
        if q == 40:
            for i in range(20,27):
                u[i,:] = beta_begin[i] + np.linspace(0,n-1,n)/10
            beta[:,20] = 1.5*np.sin(20/n*np.pi*u[20,:])+2.5
            beta[:,21] = 1.5*np.cos(17/n*np.pi*u[21,:]+0.4)+2.5
            beta[:,22] = 1.5*np.sin(17/n*np.pi*u[22,:]-1.2)+2.5
            beta[:,23] = 1.5*np.cos(20/n*np.pi*u[23,:]-2)+2.5
            beta[:,24] = 1.5*np.sin(20/n*np.pi*u[24,:])+2.5
            beta[:,25] = 1.5*np.cos(17/n*np.pi*u[25,:]+0.4)+2.5
            beta[:,26] = 1.5*np.sin(17/n*np.pi*u[26,:]-1.2)+2.5

        for i in range(7,10):
            u[i,:] = beta_begin[i] + np.linspace(0,n-1,n)/100
        beta_exp3 = np.array(list(map(lambda x: math.exp(x), -1*u[7,:])))
        beta[:,7] = 4*beta_exp3+1
        beta_exp3 = np.array(list(map(lambda x: math.exp(x), -0.9*u[8,:])))
        beta[:,8] = 4*beta_exp3+1
        beta_exp3 = np.array(list(map(lambda x: math.exp(x), -0.8*u[9,:])))
        beta[:,9] = 4*beta_exp3+1
        if q == 40:
            for i in range(27,30):
                u[i,:] = beta_begin[i] + np.linspace(0,n-1,n)/100
            beta_exp3 = np.array(list(map(lambda x: math.exp(x), -1*u[28,:])))
            beta[:,27] = 4*beta_exp3+1
            beta_exp3 = np.array(list(map(lambda x: math.exp(x), -0.9*u[27,:])))
            beta[:,28] = 4*beta_exp3+1
            beta_exp3 = np.array(list(map(lambda x: math.exp(x), -0.8*u[29,:])))
            beta[:,29] = 4*beta_exp3+1

        
        for i in range(10,14):
            u[i,:] = beta_begin[i] + np.linspace(0,n-1,n)/20
        beta_ln = np.array(list(map(lambda x: math.log(x), u[10,:]**3)))
        beta[:,10] = 0.5*beta_ln+1
        beta_ln = np.array(list(map(lambda x: math.log(x), u[11,:]**2.9)))
        beta[:,11] = 0.5*beta_ln+1
        beta_ln = np.array(list(map(lambda x: math.log(x), u[12,:]**2.7)))
        beta[:,12] = 0.5*beta_ln+1
        beta_ln = np.array(list(map(lambda x: math.log(x), u[13,:]**2.5)))
        beta[:,13] = 0.5*beta_ln+1
        if q == 40:
            for i in range(30,34):
                u[i,:] = beta_begin[i] + np.linspace(0,n-1,n)/20
            beta_ln = np.array(list(map(lambda x: math.log(x), u[30,:]**3)))
            beta[:,30] = 0.5*beta_ln+1
            beta_ln = np.array(list(map(lambda x: math.log(x), u[31,:]**2.9)))
            beta[:,31] = 0.5*beta_ln+1
            beta_ln = np.array(list(map(lambda x: math.log(x), u[32,:]**2.7)))
            beta[:,32] = 0.5*beta_ln+1
            beta_ln = np.array(list(map(lambda x: math.log(x), u[33,:]**2.5)))
            beta[:,33] = 0.5*beta_ln+1

            
        
        for i in range(14,20):
            u[i,:] = beta_begin[i] + np.linspace(0,n-1,n)/10
        beta[:,14] = 0.16*u[14,:]+2
        beta[:,15] = 0.15*u[15,:]+2
        beta[:,16] = 0.14*u[16,:]+2
        beta[:,17] = 0.13*u[17,:]+2
        beta[:,18] = 0.16*u[18,:]+2
        beta[:,19] = 0.15*u[19,:]+2
        if q == 40:
            for i in range(34,40):
                u[i,:] = beta_begin[i] + np.linspace(0,n-1,n)/10
            beta[:,34] = 0.16*u[34,:]+2
            beta[:,35] = 0.15*u[35,:]+2
            beta[:,36] = 0.14*u[36,:]+2
            beta[:,37] = 0.13*u[37,:]+2
            beta[:,38] = 0.16*u[38,:]+2
            beta[:,39] = 0.15*u[39,:]+2

   
    return beta,beta_begin
    


       
def simuGenerator(n,p,rho,beta):
    sigma = np.zeros(p*p).reshape(p,p)
    for i in range(p):
        for j in range(p):
            sigma[i][j]=rho**abs(i-j)
    
    X = np.random.multivariate_normal(np.array([0]*p),sigma,n)
    Y = (X*beta).sum(axis = 1) + np.random.randn(n)

    return Y, X



beta, beta_begin = gen_beta(200,500,20,'sin')
Y,X = simuGenerator(200,500,0.3)

#np.savetxt('beta.csv',beta,delimiter=',',fmt='%10.5f')
#np.savetxt('X.csv',X,delimiter=',',fmt='%10.5f')
#np.savetxt('Y.csv',Y,delimiter=',',fmt='%10.5f')


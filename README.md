# HALIC
This is a `Python` implementation of the following paper:
Luo Z, Zhang Y, Sun Y. A Penalization Method for Estimating Heterogeneous Covariate Effects in Cancer Genomic Data. Genes (Basel). 2022 Apr 15;13(4):702. doi: 10.3390/genes13040702. PMID: 35456506; PMCID: PMC9025588.

# Introduction

In this code, we consider a scenario where the effects of covariates change smoothly across subjects, which are ordered by a known auxiliary variable. We develop a penalization-based approach, which applies a penalization technique to simultaneously select important covariates and estimate their unique effects on the outcome variables of each subject.


# Demo
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
tp = sum(sig_index < sum(np.apply_along_axis(norm2, 0, beta)!=0))
fp = len(sig_index) - tp

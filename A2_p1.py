# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import csv
import timeit
from scipy.sparse import csr_matrix
from scipy.stats import norm

""" 
Part a and b
Compute CDF and pdf
use monte carlo and distribution function
"""
def f_2():
    rv = np.random.normal(1, 0.2861, 1)
    return rv

def fnX(n):
    if n==0:
        return 1
    else:
        rv = np.random.normal(0, 0.01, 1)
        xnm = fnX(n-1)
        return (xnm + rv)**2 + (xnm + rv) - 1


nsamp = 50000
v_1 = np.zeros(nsamp)

v_2 = np.random.normal(1, 0.2861, nsamp)

for ii in range(nsamp):
    v_1[ii] = fnX(3)

count, bins_count = np.histogram(v_1, bins=100)
pdf_1 = count / sum(count)
cdf_1 = np.cumsum(pdf_1)
count_2, bins_count_2 = np.histogram(v_2, bins=100)
pdf_2 = count_2 / sum(count_2)
cdf_2 = np.cumsum(pdf_2)
x_axis = bins_count[1:]

probability_cdf = norm.cdf(x_axis, loc=1, scale=0.2861)

# plt.plot(bins_count[1:], pdf, color="red", label="PDF")
plt.plot(x_axis, cdf_1, label="CDF MC")
plt.plot(x_axis, probability_cdf, label="CDF normal approx", linestyle='dashdot')
plt.grid(True)
plt.legend()
plt.show()
psum = 0
for ii in range(len(cdf_1)):
    if x_axis[ii]>=1.5:
        psum += pdf_1[ii]

print('prob greate 1.5 is', psum, 1-psum)

plt.figure()
count, bins, ignored = plt.hist(v_1, 100, density=True, label="PDF")
plt.grid(True)
plt.legend()
plt.show()



probability_cdf = norm.cdf(1.5, loc=1, scale=0.2861)
print('Probability density confidence interval greater than 5', 1-probability_cdf)

"""
Part C, markov chains
Ant in hyper cube
"""
# Prob matrix for ant
def f_markov(x):
    sv = x[:, 0]+x[:, 1]+x[:, 2]
    sv = sv/np.sum(sv)
    return sv

centroid = np.array([[0, 0, 0],
                     [1, 0, 0],
                     [1, 1, 0],
                     [0, 1, 0],
                     [0, 0, 1],
                     [1, 0, 1],
                     [1, 1, 1],
                     [0, 1, 1]])
nv = 8
connlist = {}

for ii in range(nv):
    dmy = []
    for jj in range(nv):
        if ii != jj:
            dist = np.linalg.norm(centroid[jj, :] - centroid[ii, :])
            if dist == 1: dmy.append(jj)
    connlist[ii] = dmy

Amat = np.zeros((nv, nv))
for ii in range(nv):
    sv = f_markov(centroid[connlist[ii], :])
    # for jj in connlist[ii]:
    Amat[ii, connlist[ii]] = sv

pi = np.ones(nv)/nv
# pi[0] = 1
ns = 10
Stran = np.zeros((ns+1, nv))
Stran[0, :] = pi
for ii in range(ns):
    Stran[ii+1, :] = np.dot(Stran[ii, :], Amat)

print(Stran[ns-1,:])
# Plot from here
sig = np.linspace(0, ns, ns+1)
clrs = ['b', 'g', 'r', 'm']
lsty = ['solid', 'dashdot', 'dotted']
fig, ax = plt.subplots()
kk = 0
for ii in range(int(nv)):
    kk = ii-4 if ii >= nv/2 else ii
    ax.plot(sig, Stran[:, ii], color=clrs[kk], label='point  %d'%(ii+1), linestyle=lsty[int(ii/3)])
ax.set_xlabel('Transition Steps')
ax.set_ylabel('Probability')
plt.legend() #loc='lower right'
plt.show()

"""
Part d method of moments
"""

data = np.array([1.06,	0.04,	1.18,	0.21,
                 1.36,	0.05,	0.24,	0.99,
                 2.12,	0.06,	6,	0.71,
                 0.3,	0.72,	0.02,	0.52,
                 0.47,	1.57,	0.37,	2.28])
data2 = data**2
mom1 = np.mean(data)
mom2 = np.mean(data2)
lnmn = np.mean(np.log(data))
print(mom1, mom2, mom2-(mom1**2), np.var(data))

alp_mom = (mom1**2)/(mom2 - (mom1**2))
lam_mom = (mom1)/(mom2 - (mom1**2))
print('Mom alpha and lambda ', alp_mom, lam_mom)
nab_g1 = lambda x, y: np.array([(x**2 + y), -x])/(x**2 - y)**2
nab_g2 = lambda x, y: np.array([(2*x*y), -(x**2)])/(x**2 - y)**2

var_d = np.array([np.var(data), np.var(data**2)])

sig_1 = np.dot(var_d, nab_g1(mom1, mom2)**2)
sig_2 = np.dot(var_d, nab_g2(mom1, mom2)**2)
print('Variance and standard deviations', sig_1,
      sig_2, np.sqrt(sig_1), np.sqrt(sig_2))
print('CI mom is ', 1.96*np.sqrt(sig_1)/np.sqrt(20),
      1.96*np.sqrt(sig_2)/np.sqrt(20))

# Compare MOM with actual expectation
# Plot error norm between both distributions with time.
alf_m = lambda x: np.log(x/mom1) + lnmn - psi(x)

root = fsolve(alf_m, [0.5])
print('root alpha from fsolve is ', root)
print('alpha is ', root[0])
print('lambda is ', root[0]/mom1)
alp_mle = root[0]
lam_mle = root[0]/mom1

x = np.linspace(0.001, 0.75, 50)
vals = alf_m(x)

fig, ax = plt.subplots()
ax.plot(x, vals, color='b', label='L2-norm MCS vs CDF')
ax.legend()
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
ax.set_xlabel('time')
ax.set_ylabel('Norm')
fig.show()
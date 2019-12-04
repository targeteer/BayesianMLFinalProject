import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import csv
import sklearn
from scipy import stats
import statsmodels
from statsmodels.tsa.stattools import adfuller
import numpy as np
import GPy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *



########################
## DATA VISUALIZATION ##
########################

# 1,2,3
def branin(xx, a=1, b=((5.1/(4*np.power(np.pi, 2)))), c=(5/np.pi), r=6, s=10, t=1/(8*np.pi)):
    x1 = xx[0]
    x2 = xx[1]
    term1 = a * np.power((x2 - np.multiply(b,np.power(x1, 2)) + np.multiply(c,x1) - r),2)
    term2 = s*(1-t)*np.cos(x1)
    y = term1 + term2 + s

    return y

# X = [−5, 10] × [0, 15]
x1 = np.arange(-5, 10, 15/1000)
x2 = np.arange(0, 15, 15/1000)

x1 = np.repeat(x1, repeats = 1000)
x2 = np.tile(x2, reps = 1000)

xx = [x1, x2]
y = branin(xx)

# (1) Make a heatmap of the value of the Branin function over the domain X = [−5, 10] × [0, 15]
# using a dense grid of values, with 1000 values per dimension, forming a 1000 × 1000 image.
df = pd.DataFrame.from_dict(np.array([x1,x2,y]).T)
df.columns = ['x1', 'x2', 'y']
data_pivoted = df.pivot("x1", "x2", "y")
ax = sns.heatmap(data_pivoted)
plt.show()

# (2) Describe the behavior of the function. Does it appear stationary? (That is, does the behavior
# of the function appear to be relatively constant throughout the domain?)
# DK : check https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html?highlight=adf#statsmodels.tsa.stattools.adfuller
# DK : don't run this if your computer memory is filled up. y has 1,000,000(1000*1000) points and it freezes up your computer!...PS I could never run this on my computer btw..
# stat_check = statsmodels.tsa.stattools.adfuller(df.y)
# print(stat_check[0])
# Data is not stationary because it does not have a constant trend, but rather has uphils and downhils.


# (3) Can you find a transformation of the data that makes it more stationary?



# # (4) Make a kernel density estimate of the distribution of the values for the lda and svm benchmarks.
# # kernel density estimate of LDA
# lda = pd.read_csv("lda.csv")
# dy = lda.iloc[:,3]
# # y = list(range(len(dy)))
# # y = pd.Series(y)

# plt.figure(1)
# plt.title('Kernel Density Estimate of LDA')
# plt.xlabel('y')
# plt.ylabel('Density')
# ax = dy.plot.kde()

# plt.figure(2)
# log_dy = np.log(dy)
# plt.title('Kernel Density Estimate of log LDA')
# plt.xlabel('log y')
# plt.ylabel('Density')
# ax = log_dy.plot.kde()
# plt.show()

# # kernel density estimate of SVM
# svm = pd.read_csv("svm.csv")
# dy = svm.iloc[:,3]

# ax = dy.plot.kde()
# plt.title('Kernel Density Estimate of SVM')
# plt.xlabel('y')
# plt.ylabel('Density')
# plt.show()

# (5) Again, can you nd a transformation that makes the performance better behaved?


# ###################
# ## Model fitting ##
# # ###################
# import sobol_seq
# import GPy

# # (1) Select a set of 32 training points for the Branin function in the domain X = [−5, 10] × [0, 15] using a Sobol sequence
# sobel_numbers = sobol_seq.i4_sobol_generate(2, 32)

# xx1 = sobel_numbers[:,0]
# xx2 = sobel_numbers[:,1]
# xx1_domain = xx1 * 15 - 5
# xx2_domain = xx2 * 15


# xx = list(zip(xx1_domain, xx2_domain))
# y = branin(xx)

# print(y)


# # xx: 2-dimensional input, y : 1-d output
# # xx = np.column_stack((xx1_domain,xx2_domain))
# # input = [xx1_domain,xx2_domain]




# # (2) Fit a Gaussian process model to the data using a constant mean and a squared exponential covariance

# # add noise to y
# noise = np.random.normal(0,0.001,len(y))
# y = y + noise
# y = y.reshape(-1, 1) 
# # or  y = pd.DataFrame(y)
# ## use reshape(-1, 1) if it is a single feature
# # xx1 = xx1.reshape(-1, 1)

# # squared exponential covariance
# # d = 2
# # var = 0.2 # variance
# # theta = np.asarray([0.2, 0.5, 1., 2., 4.])
# # for t in theta:
# #     se_kernel['.*lengthscale'] = t
# #     se_kernel.plot()
# #     plt.show()
# ## Larger the theta(lengthscale) is, bigger the kernel region is.
# ## Change in variance does not change the region of the kernel

# kernel = RBF(1,1)
# gpr = GaussianProcessRegressor(kernel = kernel)
# # xx = (np.array([xx1_domain, xx2_domain])).T
# gpr.fit(input,y)



# df = pd.DataFrame.from_dict(np.array([xx1_domain,xx2_domain,y]).T)
# df.columns = ['x1', 'x2', 'y']
# data_pivoted = df.pivot("x1", "x2", "y")
# ax = sns.heatmap(data_pivoted)
# plt.show()




# # plt.figure(1)
# # plt.plot(xx1_domain, y,'o', ms=5, label = "observations")
# # y_gpr, y_std = gpr.predict(xx1_domain, return_std=True)
# # plt.plot(xx1_domain, y_gpr, color='black', lw=0.2, label='predictions')
# # plt.fill_between(xx1_domain, y_gpr - 1.96*y_std, y_gpr + 1.96*y_std, color='darkorange')
# # plt.title('Guassian process using constant mean and rbf kernel')
# # plt.show()
# # print(m)

# #optimize
# # m.optimize()
# # m.plot()
# # plt.show()
# # print(m)




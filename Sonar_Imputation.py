import matplotlib.pyplot as plt
import numpy as np
import Gen_sonar as s
import torch
import inverse_utils
import dip_utils
import time
import pywt
import scipy.fftpack as spfft
from scipy.interpolate import interp1d


#NETWORK SETUP
LR = 1e-4 # learning rate
MOM = 0.9 # momentum
NUM_ITER = 7000 # number iterations
WD = 1 # weight decay for l2-regularization
Z_NUM = 32 # input seed
NGF = 64 # number of filters per layer
nc = 1 #num channels in the net I/0
alpha = 1e-5 #learning rate of Lasso
alpha_tv = 1e-1 #TV parameter for net loss
LENGTH = 1024


#CUDA SETUP
CUDA = torch.cuda.is_available()
print("On GPU: ", CUDA)

if CUDA :
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor


#SIGNAL GENERATION
signal = s.gen_data(LENGTH, 0.5, 1)

target_range = np.array(range(400, 430)) #the range in the signal where the target appears
target_range = np.union1d(target_range, np.array(range(800, 830))) #the range in the signal where the target appears
target_len = len(list(target_range))

signal[target_range] += 0.04 #add artificial target to the background signal

signal += np.random.normal(loc = 0, scale = 0.0075, size=LENGTH) #add background noise
signal = inverse_utils.normalise(signal) #normalise signal to range [-1, 1]

x = np.zeros((LENGTH, 1)) #make a holder for the signal with proper shape
x[:, 0] = signal

"""
plt.figure()
plt.plot(range(LENGTH), signal)
plt.title("Original Signal")
plt.show()
"""


#IMPUTATION SETUP
missing1 = range(375, 425) #Define the ranges of samples to drop
missing2 = range(810, 900)
missing3 = range(200, 230)

missing_samples = np.array(missing1) #unify all the missing sample ranges into one for indexing
missing_samples = np.union1d(missing_samples, np.array(missing2))
missing_samples = np.union1d(missing_samples, np.array(missing3))

kept_samples = [x for x in range(LENGTH) if x not in missing_samples] #define the sample indices which are not dropped

A = np.identity(LENGTH)[kept_samples, :] #the matrix which encodes the dropped samples when it left multiplies x

y = x[kept_samples] #the signal which we can see after samples have been dropped - equivalently y = Ax


#DIP imputation - run DIP to fill in missing samples
x_hat = dip_utils.run_DIP_short(A, y, dtype, NGF = NGF, LR=LR, MOM=MOM, WD=WD, output_size=LENGTH, num_measurements=len(kept_samples), CUDA=CUDA, num_iter=NUM_ITER, alpha_tv=alpha_tv)


#Lasso imputation
phi = spfft.idct(np.identity(LENGTH), norm='ortho', axis=0)[kept_samples, :] #define an IDCT transformation on A for use in LASSO imputation
x_hat_lasso = inverse_utils.run_Lasso(A=phi, y=y, output_size=LENGTH, alpha=1e-5)


#Gaussian imputation
x_hat_gaussian = x.copy()

surrounding1 = np.union1d(np.array(range(355, 375)), np.array(range(425, 445))) #define ranges of samples surrounding missing regions to derive signal statistics for imputing
surrounding2 = np.union1d(np.array(range(790, 810)), np.array(range(900, 920)))
surrounding3 = np.union1d(np.array(range(180, 200)), np.array(range(230, 250)))

x_hat_gaussian[missing1] = np.random.normal(np.mean(x[surrounding1]), np.std(x[surrounding1]), (len(missing1), 1)) #fill in the missing regions with Gaussian estimates informed by surrounding regions
x_hat_gaussian[missing2] = np.random.normal(np.mean(x[surrounding2]), np.std(x[surrounding2]), (len(missing2), 1))
x_hat_gaussian[missing3] = np.random.normal(np.mean(x[surrounding3]), np.std(x[surrounding3]), (len(missing3), 1))

#x_hat_gaussian[missing_samples] = np.random.normal(np.mean(y), 0.1, (len(missing_samples), 1)) #coarse global filling


#Linear interpolation
f_interp = interp1d(x = kept_samples, y = y.squeeze() + 2, kind='linear') #define a linear interpolation function
interped = (f_interp(missing_samples) - 2) #interpolate missing samples
x_hat_interp = x.copy()
x_hat_interp[missing_samples] = interped.reshape(-1, 1) #store interpolated missing samples in the corresponding missing areas


#Signal with missing values (for visualization)
x_missing = x.copy()
x_missing[missing_samples] = None


#Plotting signals before Normalization
fig, axs = plt.subplots(3, 2)
axs[0,0].plot(range(LENGTH), x, color = 'r')
axs[0,0].set_title('Original')
axs[0,1].plot(range(LENGTH), x_missing, color = 'k')
axs[0,1].set_title('Original with Missing Values')
axs[1,0].plot(range(LENGTH), x_hat, color = 'b')
axs[1,0].set_title('DIP')
axs[1,1].plot(range(LENGTH), x_hat_lasso, color = 'g')
axs[1,1].set_title('Lasso')
axs[2,0].plot(range(LENGTH), x_hat_gaussian, color = 'c')
axs[2,0].set_title('Gaussian')
axs[2,1].plot(range(LENGTH), x_hat_interp, color = 'm')
axs[2,1].set_title('Linear Interpolation')
fig.suptitle("Imputed Signals without Normalization")
plt.show()


#Normalize imputed signals - adding two to input signals to avoid issue with exploding quotients
orig_normalised = s.two_pass_filtering(x+2, 20, 35, 1)
DIP_normalised = s.two_pass_filtering(x_hat+2, 20, 35, 1)
lasso_normalised = s.two_pass_filtering(x_hat_lasso + 2, 20, 35, 1)
gaussian_normalised = s.two_pass_filtering(x_hat_gaussian + 2, 20, 35, 1)
interp_normalised = s.two_pass_filtering(x_hat_interp + 2, 20, 35, 1)


fig, axs = plt.subplots(3, 2)
axs[0,0].plot(range(LENGTH), orig_normalised, color = 'r')
axs[0,0].set_title('Original')
axs[0,1].plot(range(LENGTH), orig_normalised, color = 'r')
axs[0,1].set_title('Original')
axs[1,0].plot(range(LENGTH), DIP_normalised, color = 'b')
axs[1,0].set_title('DIP')
axs[1,1].plot(range(LENGTH), lasso_normalised, color = 'g')
axs[1,1].set_title('Lasso')
axs[2,0].plot(range(LENGTH), gaussian_normalised, color = 'c')
axs[2,0].set_title('Gaussian')
axs[2,1].plot(range(LENGTH), interp_normalised, color = 'm')
axs[2,1].set_title('Linear Interpolation')
fig.suptitle("Imputed Signals with Normalization")
plt.show()

"""
#plot signals after normalization
plt.figure()
plt.plot(range(LENGTH), orig_normalised, label = "Original", color = 'r')
plt.plot(range(LENGTH), DIP_normalised, label = "DIP", color = 'b')
plt.plot(range(LENGTH), lasso_normalised, label = "Lasso", color = 'g')
plt.plot(range(LENGTH), gaussian_normalised, label = "Gaussian", color = 'c')
plt.plot(range(LENGTH), interp_normalised, label = "Linear Interpolation", color = 'm')
plt.ylim(0.5, 1.5)
plt.legend()
plt.grid(True)
plt.show()
"""

#import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import gpflow
from scipy import stats
from sklearn.externals import joblib
# -------------------------------------------------------------------------------------------
# ----------------------section 0 Load Data-----------------------------------------
# -------------------------------------------------------------------------------------------
fxData = pd.read_csv('Xng.txt')
n = len(fxData.EURCHF)
LogR_EURUSD_full = [np.log(fxData.EURUSD[i+1])-np.log(fxData.EURUSD[i]) for i in range(0, n-1)]
LogR_EURCHF_full = [np.log(fxData.EURCHF[i+1])-np.log(fxData.EURCHF[i]) for i in range(0, n-1)]
LogR_EURNOK_full = [np.log(fxData.EURNOK[i+1])-np.log(fxData.EURNOK[i]) for i in range(0, n-1)]
LogR_EURSEK_full = [np.log(fxData.EURSEK[i+1])-np.log(fxData.EURSEK[i]) for i in range(0, n-1)]

# -------------------------------------------------------------------------------------------
# ----------------------section 1 Gaussian Process-----------------------------------------
# -------------------------------------------------------------------------------------------

# --------------------------------Function-------------------------------------------------
# 1. Normalize data
def Normalization(x,y):
    x_full = x
    x_full = np.asarray(x_full)
    x_full = x_full.reshape(len(x_full),1)
    y_full = y
    y_full = np.asarray(y_full)
    y_full = y_full.reshape(len(y_full),1)
    y1_full = np.log(y_full+0.1) #offset by 0.1
    y_full = (y1_full - np.mean(y1_full))/np.std(y1_full) #normalization
    return x_full, y_full, y1_full

# 2. Compute short term realized vol
def Vol(y,vol_window):
    Asset_vol = []
    for window_number in range(0, len(y)):
        Asset_temp = []
        Asset_temp = y[window_number:vol_window + window_number]
        Asset_temp = np.sqrt(sum(np.square(Asset_temp)) / len(Asset_temp))
        Asset_vol = np.append(Asset_vol, Asset_temp)
    return Asset_vol

# 3. Univariate Autoregressive GP
def UniAutoGP(vol,start_from,training_windowno,testing_windowno,window_size,prediction_no):
    x_training = []
    for i in range(0,training_windowno):
        x_temp = vol[i+start_from:i+window_size+start_from]
        x_training = np.append(x_training,x_temp)
    x_training = x_training.reshape(training_windowno,window_size)
    #x training input: 1000 x 50

    y_training = []
    for i in range(0,training_windowno):
        y_temp = vol[i+window_size+start_from+prediction_no]
        y_training = np.append(y_training,y_temp)
    y_training = y_training.reshape(training_windowno,1)
    #y training input: 1000 x 1

    x_testing = []
    for i in range(0,testing_windowno):
        x_temp = vol[i+training_windowno+start_from+prediction_no:i+window_size+training_windowno+start_from+prediction_no]
        x_testing = np.append(x_testing,x_temp)
    x_testing = x_testing.reshape(training_windowno,window_size)
    #x testing: 1000 x 1000

    m = gpflow.models.GPR(x_training, y_training, kern=gpflow.kernels.Matern32(1))

    lengthscale = 10
    m.kern.lengthscales = lengthscale
    m.kern.lengthscales.trainable = False

    gpflow.train.ScipyOptimizer().minimize(m)
    m.as_pandas_table()

    mean, var = m.predict_y(x_testing)

    return mean, var

# 4. Multivariate Autoregressive GP
def MultiAutoGP(vol_asset1, vol_USD, start_from, training_windowno,testing_windowno, window_size,prediction_no):
    x_training_multi = []

    for i in range(0, training_windowno):
        x_temp_asset1 = vol_asset1[i+start_from:i + window_size+start_from]
        x_temp_asset2 = vol_USD[i+start_from:i + window_size+start_from]
        x_temp_multi = np.append(x_temp_asset2, x_temp_asset2)
        x_training_multi = np.append(x_training_multi, x_temp_multi)
    x_training_multi = x_training_multi.reshape(training_windowno, 2 * window_size)

    y_training = []
    for i in range(0, training_windowno):
        y_temp = vol_asset1[i + window_size+start_from+prediction_no]
        y_training = np.append(y_training, y_temp)
    y_training = y_training.reshape(training_windowno, 1)

    x_testing = []
    for i in range(0, testing_windowno):
        x_temp = vol_asset1[i + training_windowno+start_from+prediction_no:i + window_size + training_windowno+start_from+prediction_no]
        x_testing = np.append(x_testing, x_temp)
    x_testing = x_testing.reshape(training_windowno,window_size)

    # Kernel
    k1 = gpflow.kernels.Matern32(input_dim=1, active_dims=[0])
    k2 = gpflow.kernels.Matern32(input_dim=1, active_dims=[1])
    k = k1 + k2
    m_multi = gpflow.models.GPR(x_training_multi, y_training, kern=k)
    lengthscale = 10
    m_multi.kern.kernels[0].lengthscales = lengthscale
    m_multi.kern.kernels[1].lengthscales = lengthscale
    m_multi.kern.kernels[0].lengthscales.trainable = False
    m_multi.kern.kernels[1].lengthscales.trainable = False
    gpflow.train.ScipyOptimizer().minimize(m_multi)
    print(m_multi.as_pandas_table())

    mean_multi, var_multi = m_multi.predict_y(x_testing)
    return mean_multi, var_multi

# 5. Plot
def PlotGP(x_full,vol,mean,var,mean_multi,var_multi,window_no,window_size,prediction_no):
    x_train = x_full[window_no:window_no+window_size]
    y_train = vol[window_no:window_no+window_size]
    x_test = x_full[window_no+window_size+prediction_no]
    y_test = vol[window_no+window_size+prediction_no]
    y_predict_mean = mean[window_no]
    y_predict_var = var[window_no]
    y_predict_mean_multi = mean_multi[window_no]
    y_predict_var_multi = var_multi[window_no]
    plt.figure(1)
    plt.subplot(211)
    plt.title('Univariate GP Prediction')
    plt.plot(x_train, y_train, 'kx', mew=2)
    plt.plot(x_test, y_test, 'bx', mew=2)
    plt.plot(x_test, y_predict_mean,'gx', mew=2)
    plt.fill_between(x_test, y_predict_mean - np.sqrt(y_predict_var), y_predict_mean + np.sqrt(y_predict_var),color='C2', alpha=0.2)

    plt.subplot(212)
    plt.title('Multivariate GP Prediction')
    plt.plot(x_train, y_train, 'kx', mew=2)
    plt.plot(x_test, y_test, 'bx', mew=2)
    plt.plot(x_test, y_predict_mean_multi,'gx', mew=2)
    plt.fill_between(x_test, y_predict_mean_multi - np.sqrt(y_predict_var_multi), y_predict_mean_multi + np.sqrt(y_predict_var_multi),color='C2', alpha=0.2)
    plt.show()

# 5. Performance Metrics
def PerformanceMetrics(vol,mean,mean_multi,start_from,window_size,testing_windowno,prediction_no):
    error_single = []
    for k in range(0, testing_windowno):
        mean_pred_single = mean[k]
        real_mean = vol[k + window_size + testing_windowno+start_from+prediction_no]
        error_temp = mean_pred_single - real_mean
        error_single = np.append(error_single, error_temp)
    metrics_single = abs(sum(error_single) / len(error_single))

    error_multi = []
    for k in range(0, testing_windowno):
        mean_pred_multi = mean_multi[k]
        real_mean = vol[k + window_size + testing_windowno+start_from+prediction_no]
        error_temp = mean_pred_multi - real_mean
        error_multi = np.append(error_multi, error_temp)
    metrics_multi = abs(sum(error_multi) / len(error_multi))
    return error_single,error_multi,metrics_single,metrics_multi
# -------------------------------------------------------------------------------------------
# ----------------------section 2 Implementation-----------------------------
# -------------------------------------------------------------------------------------------
window_size = 50 #size of each window
training_windowno = 3000
testing_windowno = 3000
prediction_no = 10
start_from = 0 #window starting position
vol_window = 10
window_no = 500

# ---------------------------------------Positive GPR-----------------------------------------------------
#
# # Data process
Positive_y_full_USD = [LogR_EURUSD_full[i] for i in range(0, n-1) if LogR_EURUSD_full[i]>0]
Positive_x_full_USD = [i*1.0 for i in range(0, n-1) if LogR_EURUSD_full[i]>0]

Positive_y_full_CHF = [LogR_EURCHF_full[i] for i in range(0, n-1) if LogR_EURCHF_full[i]>0]
Positive_x_full_CHF = [i*1.0 for i in range(0, n-1) if LogR_EURCHF_full[i]>0]

Positive_y_full_NOK = [LogR_EURNOK_full[i] for i in range(0, n-1) if LogR_EURNOK_full[i]>0]
Positive_x_full_NOK = [i*1.0 for i in range(0, n-1) if LogR_EURNOK_full[i]>0]

Positive_y_full_SEK = [LogR_EURSEK_full[i] for i in range(0, n-1) if LogR_EURSEK_full[i]>0]
Positive_x_full_SEK = [i*1.0 for i in range(0, n-1) if LogR_EURSEK_full[i]>0]


[pos_x_full_USD, pos_y_full_USD,pos_y_norm_USD] = Normalization(Positive_x_full_USD,Positive_y_full_USD)
pos_y_vol_USD = Vol(pos_y_full_USD,vol_window)

[pos_x_full_CHF, pos_y_full_CHF,pos_y_norm_CHF] = Normalization(Positive_x_full_CHF,Positive_y_full_CHF)
pos_y_vol_CHF = Vol(pos_y_full_CHF,vol_window)

[pos_x_full_NOK, pos_y_full_NOK,pos_y_norm_NOK] = Normalization(Positive_x_full_NOK,Positive_y_full_NOK)
pos_y_vol_NOK = Vol(pos_y_full_NOK,vol_window)

[pos_x_full_SEK, pos_y_full_SEK,pos_y_norm_SEK] = Normalization(Positive_x_full_SEK,Positive_y_full_SEK)
pos_y_vol_SEK = Vol(pos_y_full_SEK,vol_window)
# Gaussian Process:

#CHF
mean_CHF, var_CHF = UniAutoGP(pos_y_vol_CHF, start_from, training_windowno, testing_windowno, window_size,prediction_no)
mean_multi_CHF, var_multi_CHF = MultiAutoGP(pos_y_vol_CHF, pos_y_vol_USD, start_from, training_windowno,testing_windowno, window_size,prediction_no)
PlotGP(pos_x_full_CHF,pos_y_vol_CHF,mean_CHF,var_CHF,mean_multi_CHF,var_multi_CHF,window_no,window_size,prediction_no)
error_single_CHF_pos,error_multi_CHF_pos,metrics_single_CHF,metrics_multi_CHF = PerformanceMetrics(pos_y_vol_CHF,mean_CHF,mean_multi_CHF,start_from,window_size,testing_windowno,prediction_no)

print(metrics_single_CHF, 'metrics_single_CHF')
print(metrics_multi_CHF, 'metrics_multi_CHF')

#NOK
mean_NOK, var_NOK = UniAutoGP(pos_y_vol_NOK, start_from, training_windowno, testing_windowno, window_size,prediction_no)
mean_multi_NOK, var_multi_NOK = MultiAutoGP(pos_y_vol_NOK, pos_y_vol_USD, start_from, training_windowno,testing_windowno, window_size,prediction_no)
PlotGP(pos_x_full_NOK,pos_y_vol_NOK,mean_NOK,var_NOK,mean_multi_NOK,var_multi_NOK,window_no,window_size,prediction_no)
error_single_NOK_pos,error_multi_NOK_pos,metrics_single_NOK,metrics_multi_NOK = PerformanceMetrics(pos_y_vol_NOK,mean_NOK,mean_multi_NOK,start_from,window_size,testing_windowno,prediction_no)

print(metrics_single_NOK, 'metrics_single_NOK')
print(metrics_multi_NOK, 'metrics_multi_NOK')

#SEK
mean_SEK, var_SEK = UniAutoGP(pos_y_vol_SEK, start_from, training_windowno, testing_windowno, window_size,prediction_no)
mean_multi_SEK, var_multi_SEK = MultiAutoGP(pos_y_vol_SEK, pos_y_vol_USD, start_from, training_windowno,testing_windowno, window_size,prediction_no)
PlotGP(pos_x_full_SEK,pos_y_vol_SEK,mean_SEK,var_SEK,mean_multi_SEK,var_multi_SEK,window_no,window_size,prediction_no)
error_single_SEK_pos,error_multi_SEK_pos, metrics_single_SEK,metrics_multi_SEK = PerformanceMetrics(pos_y_vol_SEK,mean_SEK,mean_multi_SEK,start_from,window_size,testing_windowno,prediction_no)

print(metrics_single_SEK, 'metrics_single_SEK')
print(metrics_multi_SEK, 'metrics_multi_SEK')

joblib.dump(error_single_CHF_pos,'error_single_CHF_pos')
joblib.dump(error_multi_CHF_pos,'error_multi_CHF_pos')
joblib.dump(error_single_NOK_pos,'error_single_NOK_pos')
joblib.dump(error_multi_NOK_pos,'error_multi_NOK_pos')
joblib.dump(error_single_SEK_pos,'error_single_SEK_pos')
joblib.dump(error_multi_SEK_pos,'error_multi_SEK_pos')



# # ---------------------------------------Negative GPR-----------------------------------------------------
# #
# # Data process
# Negative_y_full_USD = [-LogR_EURUSD_full[i] for i in range(0, n-1) if LogR_EURUSD_full[i]<0]
# Negative_x_full_USD = [i*1.0 for i in range(0, n-1) if LogR_EURUSD_full[i]<0]
#
# Negative_y_full_CHF = [-LogR_EURCHF_full[i] for i in range(0, n-1) if LogR_EURCHF_full[i]<0]
# Negative_x_full_CHF = [i*1.0 for i in range(0, n-1) if LogR_EURCHF_full[i]<0]
#
# Negative_y_full_NOK = [-LogR_EURNOK_full[i] for i in range(0, n-1) if LogR_EURNOK_full[i]<0]
# Negative_x_full_NOK = [i*1.0 for i in range(0, n-1) if LogR_EURNOK_full[i]<0]
#
# Negative_y_full_SEK = [-LogR_EURSEK_full[i] for i in range(0, n-1) if LogR_EURSEK_full[i]<0]
# Negative_x_full_SEK = [i*1.0 for i in range(0, n-1) if LogR_EURSEK_full[i]<0]
#
#
# [neg_x_full_USD, neg_y_full_USD,neg_y_norm_USD] = Normalization(Negative_x_full_USD,Negative_y_full_USD)
# neg_y_vol_USD = Vol(neg_y_full_USD,vol_window)
#
# [neg_x_full_CHF, neg_y_full_CHF,neg_y_norm_CHF] = Normalization(Negative_x_full_CHF,Negative_y_full_CHF)
# neg_y_vol_CHF = Vol(neg_y_full_CHF,vol_window)
#
# [neg_x_full_NOK, neg_y_full_NOK,neg_y_norm_NOK] = Normalization(Negative_x_full_NOK,Negative_y_full_NOK)
# neg_y_vol_NOK = Vol(neg_y_full_NOK,vol_window)
#
# [neg_x_full_SEK, neg_y_full_SEK,neg_y_norm_SEK] = Normalization(Negative_x_full_SEK,Negative_y_full_SEK)
# neg_y_vol_SEK = Vol(neg_y_full_SEK,vol_window)
# # Gaussian Process:
#
# #CHF
# mean_CHF_neg, var_CHF_neg = UniAutoGP(neg_y_vol_CHF, start_from, training_windowno, testing_windowno, window_size,prediction_no)
# mean_multi_CHF_neg, var_multi_CHF_neg = MultiAutoGP(neg_y_vol_CHF, neg_y_vol_USD, start_from, training_windowno,testing_windowno, window_size,prediction_no)
# PlotGP(neg_x_full_CHF,neg_y_vol_CHF,mean_CHF_neg,var_CHF_neg,mean_multi_CHF_neg,var_multi_CHF_neg,window_no,window_size,prediction_no)
# error_single_CHF_neg,error_multi_CHF_neg,metrics_single_CHF_neg,metrics_multi_CHF_neg = PerformanceMetrics(neg_y_vol_CHF,mean_CHF_neg,mean_multi_CHF_neg,start_from,window_size,testing_windowno,prediction_no)
#
# print(metrics_single_CHF_neg, 'metrics_single_CHF_neg')
# print(metrics_multi_CHF_neg, 'metrics_multi_CHF_neg')
#
# #NOK
# mean_NOK_neg, var_NOK_neg = UniAutoGP(neg_y_vol_NOK, start_from, training_windowno, testing_windowno, window_size,prediction_no)
# mean_multi_NOK_neg, var_multi_NOK_neg = MultiAutoGP(neg_y_vol_NOK, neg_y_vol_USD, start_from, training_windowno,testing_windowno, window_size,prediction_no)
# PlotGP(neg_x_full_NOK,neg_y_vol_NOK,mean_NOK_neg,var_NOK_neg,mean_multi_NOK_neg,var_multi_NOK_neg,window_no,window_size,prediction_no)
# error_single_NOK_neg,error_multi_NOK_neg,metrics_single_NOK_neg,metrics_multi_NOK_neg = PerformanceMetrics(neg_y_vol_NOK,mean_NOK_neg,mean_multi_NOK_neg,start_from,window_size,testing_windowno,prediction_no)
#
# print(metrics_single_NOK_neg, 'metrics_single_NOK_neg')
# print(metrics_multi_NOK_neg, 'metrics_multi_NOK_neg')
#
# #SEK
# mean_SEK_neg, var_SEK_neg = UniAutoGP(neg_y_vol_SEK, start_from, training_windowno, testing_windowno, window_size,prediction_no)
# mean_multi_SEK_neg, var_multi_SEK_neg = MultiAutoGP(neg_y_vol_SEK, neg_y_vol_USD, start_from, training_windowno,testing_windowno, window_size,prediction_no)
# PlotGP(neg_x_full_SEK,neg_y_vol_SEK,mean_SEK_neg,var_SEK_neg,mean_multi_SEK_neg,var_multi_SEK_neg,window_no,window_size,prediction_no)
# error_single_SEK_neg,error_multi_SEK_neg,metrics_single_SEK_neg,metrics_multi_SEK_neg = PerformanceMetrics(neg_y_vol_SEK,mean_SEK_neg,mean_multi_SEK_neg,start_from,window_size,testing_windowno,prediction_no)
#
# print(metrics_single_SEK_neg, 'metrics_single_SEK_neg')
# print(metrics_multi_SEK_neg, 'metrics_multi_SEK_neg')
#
# # print(np.shape(error_single_SEK_neg))
#
# joblib.dump(error_single_CHF_neg,'error_single_CHF_neg')
# joblib.dump(error_multi_CHF_neg,'error_multi_CHF_neg')
# joblib.dump(error_single_NOK_neg,'error_single_NOK_neg')
# joblib.dump(error_multi_NOK_neg,'error_multi_NOK_neg')
# joblib.dump(error_single_SEK_neg,'error_single_SEK_neg')
# joblib.dump(error_multi_SEK_neg,'error_multi_SEK_neg')
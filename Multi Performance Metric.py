#import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import gpflow
from arch import arch_model
from scipy import stats
from sklearn.externals import joblib
import scipy


# -------------------------------------------------------------------------------------------
# ----------------------section 0 Load Data-----------------------------------------
# -------------------------------------------------------------------------------------------
fxData = pd.read_csv('Xng.txt')
n = len(fxData.EURUSD)
LogR_EURUSD_full = [np.log(fxData.EURUSD[i+1])-np.log(fxData.EURUSD[i]) for i in range(0, n-1)]
# LogR_EURCHF_full = [np.log(fxData.EURCHF[i+1])-np.log(fxData.EURCHF[i]) for i in range(0, n-1)]
# LogR_EURNOK_full = [np.log(fxData.EURNOK[i+1])-np.log(fxData.EURNOK[i]) for i in range(0, n-1)]
LogR_EURSEK_full = [np.log(fxData.EURSEK[i+1])-np.log(fxData.EURSEK[i]) for i in range(0, n-1)]


# Normalize data
def Normalization(x,y):
    x_full = x
    x_full = np.asarray(x_full)
    x_full = x_full.reshape(len(x_full),1)
    y_full = y
    y_full = np.asarray(y_full)
    y_full = y_full.reshape(len(y_full),1)
    y_full = np.log(y_full) #offset by 0.1
    return x_full, y_full


# Compute short term realized vol
def Vol(y,vol_window):
    Asset_vol1 = []
    for window_number in range(0, len(y)):
        Asset_temp = []
        Asset_temp = y[window_number:vol_window + window_number]
        Asset_temp = np.sqrt(sum(np.square(Asset_temp)) / len(Asset_temp))
        Asset_vol1 = np.append(Asset_vol1, Asset_temp)
    Asset_vol = (Asset_vol1 - np.mean(Asset_vol1)) / np.std(Asset_vol1)
    return Asset_vol

# -------------------------------------------------------------------------------------------
# ----------------------section 1 Performance Metrics-----------------------------------------
# -------------------------------------------------------------------------------------------

# --------------------------------Function-------------------------------------------------

def OneStepError(window_no,start_from,pre_range,window_size,mean_load,y_vol):
    error = []
    for k in range(0,window_no):
        mean_window = mean_load[k]
        pred_mean = mean_window[window_size]
        real_mean = y_vol[k+window_size+start_from]
        error_temp = pred_mean-real_mean
        error = np.append(error,error_temp)
    metrics = abs(sum(error)/len(error))
    return error,metrics


def AveError(window_no,start_from,pre_range,window_size,mean_load,y_vol,method):
    error_all = []
    error_mean_matrix = []
    total_range = pre_range+window_size
    for n in range(0,window_no):
        mean_window = mean_load[n]
        real_mean = y_vol[n+window_size+start_from:n+total_range+start_from]
        error_temp = [mean_window[i+window_size]-real_mean[i] for i in range(0, pred_range)]
        error_all = np.append(error_all,error_temp)
        error_full = np.array_split(abs(error_all), window_no)
        metrics_full = abs(sum(error_all))/len(error_all)

    for k in range(0,pre_range):
        error_n = [error_full[i][k] for i in range(0,window_no)] #i = array number(window), k = index(test data no)
        error_mean_n = np.mean(error_n)
        error_mean_matrix.append(error_mean_n)
    plt.plot(error_mean_matrix)
    if method == 0:
        plt.title('Average Negative single EURSEK GPR Error')
    if method == 1:
        plt.title('Average Negative multi EURSEK GPR Error')
    if method == 2:
        plt.title('Average Negative coregionalized EURSEK GPR Error')
    plt.show()

    return error_full, metrics_full

#Box Plot
def BoxPlot(window_no,pre_range,error_full,method):
    error_box_matrix = []
    error_median = []
    for k in range(0,pre_range):
        error_box = [abs(error_full[i][k]) for i in range(0,window_no)]
        error_box_matrix.append(error_box)
        error_median.append(np.median(error_box))
    print(error_median,'error_median')
    print(np.mean(error_median),'mean median error')
    plt.boxplot(error_box_matrix,0, '',showmeans=True)
    if method == 0:
        plt.title('Box Plot of Average Negative single GPR Error')
    if method == 1:
        plt.title('Box Plot of Average Negative multi GPR Error')
    if method == 2:
        plt.title('Box Plot of Average Negative coregionalized GPR Error')
    plt.show()


# -------------------------------------------------------------------------------------------
# ----------------------Data Processing-------------------------------------
# -------------------------------------------------------------------------------------------
pred_range = 10 #predicted range
window_size = 50 #size of each window
total_range = pred_range+window_size
start_from = 0 #window starting position
vol_window = 10


# ---------------------------------------Positive GPR-----------------------------------------------------
#
# Data process

# 1. EURUSD
Positive_y_EURUSD = [LogR_EURUSD_full[i] for i in range(0, n-1) if LogR_EURUSD_full[i]>0]
Positive_x_EURUSD = [i*1.0 for i in range(0, n-1) if LogR_EURUSD_full[i]>0]
[pos_x_EURUSD, pos_y_EURUSD] = Normalization(Positive_x_EURUSD,Positive_y_EURUSD)


# # 2. EURCHF
# Positive_y_EURCHF = [LogR_EURCHF_full[i] for i in range(0, n-1) if LogR_EURCHF_full[i]>0]
# Positive_x_EURCHF = [i*1.0 for i in range(0, n-1) if LogR_EURCHF_full[i]>0]
# [pos_x_EURCHF, pos_y_EURCHF] = Normalization(Positive_x_EURCHF,Positive_y_EURCHF)
#
#
# # 2. EURNOK
# Positive_y_EURNOK = [LogR_EURNOK_full[i] for i in range(0, n-1) if LogR_EURNOK_full[i]>0]
# Positive_x_EURNOK = [i*1.0 for i in range(0, n-1) if LogR_EURNOK_full[i]>0]
# [pos_x_EURNOK, pos_y_EURNOK] = Normalization(Positive_x_EURNOK,Positive_y_EURNOK)


# 2. EURSEK
Positive_y_EURSEK = [LogR_EURSEK_full[i] for i in range(0, n-1) if LogR_EURSEK_full[i]>0]
Positive_x_EURSEK = [i*1.0 for i in range(0, n-1) if LogR_EURSEK_full[i]>0]
[pos_x_EURSEK, pos_y_EURSEK] = Normalization(Positive_x_EURSEK,Positive_y_EURSEK)


# 3. Volatility
# pos_vol_EURCHF = Vol(pos_y_EURCHF,vol_window)
pos_vol_EURUSD = Vol(pos_y_EURUSD,vol_window)
# pos_vol_EURNOK = Vol(pos_y_EURNOK,vol_window)
pos_vol_EURSEK = Vol(pos_y_EURSEK,vol_window)

# -----------------------------Positive Performance Metrics,EURNOK----------------------------------
pos_window_number = 3000


pos_mean_Single =joblib.load('mean_SEK_3000_pos_fixed.pkl')
pos_mean_Multi = joblib.load('mean_pos_EURSEK_multi3000_fixed.pkl')
pos_mean_Coreg = joblib.load('mean_pos_EURSEK_Coreg3000_fixed.pkl')



[error_single_one, Metric_single_one] = OneStepError(pos_window_number,start_from,pred_range,window_size,pos_mean_Single,pos_vol_EURSEK)
[error_multi_one, Metric_multi_one] = OneStepError(pos_window_number,start_from,pred_range,window_size,pos_mean_Multi,pos_vol_EURSEK)
[error_coreg_one, Metric_coreg_one] = OneStepError(pos_window_number,start_from,pred_range,window_size,pos_mean_Coreg,pos_vol_EURSEK)


[error_single_all, Metric_single_all] = AveError(pos_window_number,start_from,pred_range,window_size,pos_mean_Single,pos_vol_EURSEK,0)
[error_multi_all, Metric_multi_all] = AveError(pos_window_number,start_from,pred_range,window_size,pos_mean_Multi,pos_vol_EURSEK,1)
[error_coreg_all, Metric_coreg_all] = AveError(pos_window_number,start_from,pred_range,window_size,pos_mean_Coreg,pos_vol_EURSEK,2)


BoxPlot(pos_window_number,pred_range,error_single_all,0)
BoxPlot(pos_window_number,pred_range,error_multi_all,1)
BoxPlot(pos_window_number,pred_range,error_coreg_all,2)


joblib.dump(error_single_one, 'error_single_pos_3000_SEK.pkl')
joblib.dump(error_multi_one, 'error_multi_pos_3000_SEK.pkl')
joblib.dump(error_coreg_one, 'error_coreg_pos_3000_SEK.pkl')

print(Metric_single_one,'Metric_single_one')
print(Metric_multi_one,'Metric_multi_one')
print(Metric_coreg_one,'Metric_coreg_one')

print(Metric_single_all,'Metric_single_all')
print(Metric_multi_all,'Metric_multi_all')
print(Metric_coreg_all,'Metric_coreg_all')





# # -----------------------------Negative Performance Metrics----------------------------------
#
#---------------------------------------Negative GPR-----------------------------------------------------
# Data process

# # 1. EURUSD
# Negative_y_EURUSD = [-LogR_EURUSD_full[i] for i in range(0, n-1) if LogR_EURUSD_full[i]<0]
# Negative_x_EURUSD = [i*1.0 for i in range(0, n-1) if LogR_EURUSD_full[i]<0]
# [ne_x_EURUSD, neg_y_EURUSD] = Normalization(Negative_x_EURUSD,Negative_y_EURUSD)
# #
# # 2. EURCHF
# Negative_y_EURCHF = [-LogR_EURCHF_full[i] for i in range(0, n-1) if LogR_EURCHF_full[i]<0]
# Negative_x_EURCHF = [i*1.0 for i in range(0, n-1) if LogR_EURCHF_full[i]<0]
# [neg_x_EURCHF, neg_y_EURCHF] = Normalization(Negative_x_EURCHF,Negative_y_EURCHF)
# #
# #
# # 3. EURNOK
# Negative_y_EURNOK = [-LogR_EURNOK_full[i] for i in range(0, n-1) if LogR_EURNOK_full[i]<0]
# Negative_x_EURNOK = [i*1.0 for i in range(0, n-1) if LogR_EURNOK_full[i]<0]
# [neg_x_EURNOK, neg_y_EURNOK] = Normalization(Negative_x_EURNOK,Negative_y_EURNOK)
#

# # 3. EURSEK
# Negative_y_EURSEK = [-LogR_EURSEK_full[i] for i in range(0, n-1) if LogR_EURSEK_full[i]<0]
# Negative_x_EURSEK = [i*1.0 for i in range(0, n-1) if LogR_EURSEK_full[i]<0]
# [neg_x_EURSEK, neg_y_EURSEK] = Normalization(Negative_x_EURSEK,Negative_y_EURSEK)
#
# # #
# # 3. Volatility
# # neg_vol_EURCHF = Vol(neg_y_EURCHF,vol_window)
# neg_vol_EURUSD = Vol(neg_y_EURUSD,vol_window)
# # neg_vol_EURNOK = Vol(neg_y_EURNOK,vol_window)
# neg_vol_EURSEK = Vol(neg_y_EURSEK,vol_window)
# # #

# # -----------------------------Negative Performance Metrics,EURNOK----------------------------------
# neg_window_number = 3000
#
# neg_mean_Single =joblib.load('mean_SEK_3000_neg_fixed.pkl')
# neg_mean_Multi = joblib.load('mean_neg_EURSEK_multi3000_fixed.pkl')
# neg_mean_Coreg = joblib.load('mean_neg_EURSEK_coregion_3000.pkl')
#
#
# [error_single_one, Metric_single_one] = OneStepError(neg_window_number,start_from,pred_range,window_size,neg_mean_Single,neg_vol_EURSEK)
# [error_multi_one, Metric_multi_one] = OneStepError(neg_window_number,start_from,pred_range,window_size,neg_mean_Multi,neg_vol_EURSEK)
# [error_coreg_one, Metric_coreg_one] = OneStepError(neg_window_number,start_from,pred_range,window_size,neg_mean_Coreg,neg_vol_EURSEK)
#
# [error_single_all, Metric_single_all] = AveError(neg_window_number,start_from,pred_range,window_size,neg_mean_Single,neg_vol_EURSEK,0)
# [error_multi_all, Metric_multi_all] = AveError(neg_window_number,start_from,pred_range,window_size,neg_mean_Multi,neg_vol_EURSEK,1)
# [error_coreg_all, Metric_coreg_all] = AveError(neg_window_number,start_from,pred_range,window_size,neg_mean_Coreg,neg_vol_EURSEK,2)
#
#
# BoxPlot(neg_window_number,pred_range,error_single_all,0)
# BoxPlot(neg_window_number,pred_range,error_multi_all,1)
# BoxPlot(neg_window_number,pred_range,error_coreg_all,2)
#
#
# joblib.dump(error_single_one, 'error_single_neg_3000_SEK.pkl')
# joblib.dump(error_multi_one, 'error_multi_neg_3000_SEK.pkl')
# joblib.dump(error_coreg_one, 'error_coreg_neg_3000_SEK.pkl')
#
# print(Metric_single_one,'Metric_single_one')
# print(Metric_multi_one,'Metric_multi_one')
# print(Metric_coreg_one,'Metric_coreg_one')
#
# print(Metric_single_all,'Metric_single_all')
# print(Metric_multi_all,'Metric_multi_all')
# print(Metric_coreg_all,'Metric_coreg_all')
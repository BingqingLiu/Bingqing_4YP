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
LogR_EURCHF_full = [np.log(fxData.EURCHF[i+1])-np.log(fxData.EURCHF[i]) for i in range(0, n-1)]
LogR_EURNOK_full = [np.log(fxData.EURNOK[i+1])-np.log(fxData.EURNOK[i]) for i in range(0, n-1)]
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
        pred_mean = mean_load[k+start_from]
        real_mean = y_vol[k+window_size+start_from]
        error_temp = pred_mean-real_mean
        error = np.append(error,error_temp)
    metrics = abs(sum(error)/len(error))
    return error,metrics

# -------------------------------------------------------------------------------------------
# ----------------------Data Processing-------------------------------------
# -------------------------------------------------------------------------------------------
pred_range = 10 #predicted range
window_size = 0 #size of each window
total_range = pred_range+window_size
start_from = 0 #window starting position
vol_window = 10


# # ---------------------------------------Positive GPR-----------------------------------------------------
# #
# # Data process
#
#
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
#
#
# # 2. EURSEK
# Positive_y_EURSEK = [LogR_EURSEK_full[i] for i in range(0, n-1) if LogR_EURSEK_full[i]>0]
# Positive_x_EURSEK = [i*1.0 for i in range(0, n-1) if LogR_EURSEK_full[i]>0]
# [pos_x_EURSEK, pos_y_EURSEK] = Normalization(Positive_x_EURSEK,Positive_y_EURSEK)
#
#
# # 3. Volatility
# pos_vol_EURCHF = Vol(pos_y_EURCHF,vol_window)
# pos_vol_EURNOK = Vol(pos_y_EURNOK,vol_window)
# pos_vol_EURSEK = Vol(pos_y_EURSEK,vol_window)
#
# # -----------------------------Positive Performance Metrics,EURNOK----------------------------------
# pos_window_number = 3000
#
#
# pos_mean_CHF =joblib.load('GARCH3000pos_EURCHF_new.pkl')
# pos_mean_NOK = joblib.load('GARCH3000pos_EURNOK_new.pkl')
# pos_mean_SEK = joblib.load('GARCH3000pos_EURSEK_new.pkl')
#
#
#
# [GARCH_error_pos_CHF, Metric_pos_CHF] = OneStepError(pos_window_number,start_from,pred_range,window_size,pos_mean_CHF,pos_vol_EURCHF)
# [GARCH_error_pos_NOK, Metric_pos_NOK] = OneStepError(pos_window_number,start_from,pred_range,window_size,pos_mean_NOK,pos_vol_EURNOK)
# [GARCH_error_pos_SEK, Metric_pos_SEK] = OneStepError(pos_window_number,start_from,pred_range,window_size,pos_mean_SEK,pos_vol_EURSEK)
#
# joblib.dump(GARCH_error_pos_CHF, 'error_GARCH_pos_3000_CHF.pkl')
# joblib.dump(GARCH_error_pos_NOK, 'error_GARCH_pos_3000_NOK.pkl')
# joblib.dump(GARCH_error_pos_SEK, 'error_GARCH_pos_3000_SEK.pkl')
#
# print(Metric_pos_CHF,'Metric_pos_CHF')
# print(Metric_pos_NOK,'Metric_pos_NOK')
# print(Metric_pos_SEK,'Metric_pos_SEK')



# # -----------------------------Negative Performance Metrics----------------------------------
#
#---------------------------------------Negative GPR-----------------------------------------------------
# Data process

#
# 1. EURCHF
Negative_y_EURCHF = [-LogR_EURCHF_full[i] for i in range(0, n-1) if LogR_EURCHF_full[i]<0]
Negative_x_EURCHF = [i*1.0 for i in range(0, n-1) if LogR_EURCHF_full[i]<0]
[neg_x_EURCHF, neg_y_EURCHF] = Normalization(Negative_x_EURCHF,Negative_y_EURCHF)
#
#
# 2. EURNOK
Negative_y_EURNOK = [-LogR_EURNOK_full[i] for i in range(0, n-1) if LogR_EURNOK_full[i]<0]
Negative_x_EURNOK = [i*1.0 for i in range(0, n-1) if LogR_EURNOK_full[i]<0]
[neg_x_EURNOK, neg_y_EURNOK] = Normalization(Negative_x_EURNOK,Negative_y_EURNOK)


# 3. EURSEK
Negative_y_EURSEK = [-LogR_EURSEK_full[i] for i in range(0, n-1) if LogR_EURSEK_full[i]<0]
Negative_x_EURSEK = [i*1.0 for i in range(0, n-1) if LogR_EURSEK_full[i]<0]
[neg_x_EURSEK, neg_y_EURSEK] = Normalization(Negative_x_EURSEK,Negative_y_EURSEK)

# #
# 3. Volatility
neg_vol_EURCHF = Vol(neg_y_EURCHF,vol_window)
neg_vol_EURNOK = Vol(neg_y_EURNOK,vol_window)
neg_vol_EURSEK = Vol(neg_y_EURSEK,vol_window)
# #

# -----------------------------Negative Performance Metrics,EURNOK----------------------------------
neg_window_number = 3000
#
neg_mean_CHF =joblib.load('GARCH3000neg_EURCHF.pkl')
neg_mean_NOK = joblib.load('GARCH3000neg_EURNOK.pkl')
neg_mean_SEK = joblib.load('GARCH3000neg_EURSEK.pkl')

[GARCH_error_neg_CHF, Metric_neg_CHF] = OneStepError(neg_window_number,start_from,pred_range,window_size,neg_mean_CHF,neg_vol_EURCHF)
[GARCH_error_neg_NOK, Metric_neg_NOK] = OneStepError(neg_window_number,start_from,pred_range,window_size,neg_mean_NOK,neg_vol_EURNOK)
[GARCH_error_neg_SEK, Metric_neg_SEK] = OneStepError(neg_window_number,start_from,pred_range,window_size,neg_mean_SEK,neg_vol_EURSEK)

joblib.dump(GARCH_error_neg_CHF, 'error_GARCH_neg_3000_CHF.pkl')
joblib.dump(GARCH_error_neg_NOK, 'error_GARCH_neg_3000_NOK.pkl')
joblib.dump(GARCH_error_neg_SEK, 'error_GARCH_neg_3000_SEK.pkl')

print(Metric_neg_CHF,'Metric_neg_CHF')
print(Metric_neg_NOK,'Metric_neg_NOK')
print(Metric_neg_SEK,'Metric_neg_SEK')
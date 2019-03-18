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
n = len(fxData.EURCHF)
LogR_EURCHF_full = [np.log(fxData.EURCHF[i+1])-np.log(fxData.EURCHF[i]) for i in range(0, n-1)]
LogR_EURNOK_full = [np.log(fxData.EURNOK[i+1])-np.log(fxData.EURNOK[i]) for i in range(0, n-1)]
LogR_EURSEK_full = [np.log(fxData.EURSEK[i+1])-np.log(fxData.EURSEK[i]) for i in range(0, n-1)]

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

    # y_full = np.log(y_full)
    return x_full, y_full

# 2. Compute short term realized vol
def Vol(y,vol_window):
    Asset_vol1 = []
    for window_number in range(0, len(y)):
        Asset_temp = []
        Asset_temp = y[window_number:vol_window + window_number]
        Asset_temp = np.sqrt(sum(np.square(Asset_temp)) / len(Asset_temp))
        Asset_vol1 = np.append(Asset_vol1, Asset_temp)
    Asset_vol = (Asset_vol1-np.mean(Asset_vol1))/np.std(Asset_vol1)
    return Asset_vol

# -------------------------------------------------------------------------------------------
# ----------------------Data Processing-------------------------------------
# -------------------------------------------------------------------------------------------
window_size = 50 #size of each window
start_from = 0 #window starting position
vol_window = 10
# ---------------------------------------Positive GPR-----------------------------------------------------

# Data process
Positive_y_full_EURCHF = [LogR_EURCHF_full[i] for i in range(0, n-1) if LogR_EURCHF_full[i]>0]
Positive_x_full_EURCHF = [i*1.0 for i in range(0, n-1) if LogR_EURCHF_full[i]>0]
[pos_x_full_EURCHF, pos_y_full_EURCHF] = Normalization(Positive_x_full_EURCHF,Positive_y_full_EURCHF)


Positive_y_full_EURNOK = [LogR_EURNOK_full[i] for i in range(0, n-1) if LogR_EURNOK_full[i]>0]
Positive_x_full_EURNOK = [i*1.0 for i in range(0, n-1) if LogR_EURNOK_full[i]>0]
[pos_x_full_EURNOK, pos_y_full_EURNOK] = Normalization(Positive_x_full_EURNOK,Positive_y_full_EURNOK)


Positive_y_full_EURSEK = [LogR_EURSEK_full[i] for i in range(0, n-1) if LogR_EURSEK_full[i]>0]
Positive_x_full_EURSEK = [i*1.0 for i in range(0, n-1) if LogR_EURSEK_full[i]>0]
[pos_x_full_EURSEK, pos_y_full_EURSEK] = Normalization(Positive_x_full_EURSEK,Positive_y_full_EURSEK)
# # Volatility
# pos_vol = Vol(pos_y_full,vol_window)

# ---------------------------------------Negative GPR-----------------------------------------------------
# Data process
Negative_y_full_EURCHF = [-LogR_EURCHF_full[i] for i in range(0, n-1) if LogR_EURCHF_full[i]<0]
Negative_x_full_EURCHF = [i*1.0 for i in range(0, n-1) if LogR_EURCHF_full[i]<0]
[neg_x_full_EURCHF, neg_y_full_EURCHF] = Normalization(Negative_x_full_EURCHF,Negative_y_full_EURCHF)


Negative_y_full_EURNOK = [-LogR_EURNOK_full[i] for i in range(0, n-1) if LogR_EURNOK_full[i]<0]
Negative_x_full_EURNOK = [i*1.0 for i in range(0, n-1) if LogR_EURNOK_full[i]<0]
[neg_x_full_EURNOK, neg_y_full_EURNOK] = Normalization(Negative_x_full_EURNOK,Negative_y_full_EURNOK)


Negative_y_full_EURSEK = [-LogR_EURSEK_full[i] for i in range(0, n-1) if LogR_EURSEK_full[i]<0]
Negative_x_full_EURSEK = [i*1.0 for i in range(0, n-1) if LogR_EURSEK_full[i]<0]
[neg_x_full_EURSEK, neg_y_full_EURSEK] = Normalization(Negative_x_full_EURSEK,Negative_y_full_EURSEK)

# Volatility
# neg_vol = Vol(neg_y_full,vol_window)


# -------------------------------------------------------------------------------------------
# ----------------------section 4 GARCH and comparision-----------------------------------------
# -------------------------------------------------------------------------------------------
#
# --------------------------------Function-------------------------------------------------

def GARCH_pred(window_size,window_no,y_full):
#One data prediction for 1000 windows
    window_prediction = []
    for window_number in range(0,window_no):
        y_training = y_full[window_number:window_number+window_size];#first 50 data as training data
        model = arch_model(y_training, mean='Zero', vol='GARCH', p=2, q=50)
        model_fit = model.fit()
        yhat = model_fit.forecast(horizon=1)
        result = yhat.variance.values[-1, :]
        window_prediction = np.append(window_prediction,result)
        print(window_number,window_prediction,'window_prediction')
    return window_prediction

# -------------------------------------------------------------------------------------------
# ----------------------section 4 Implementation-----------------------------
# -------------------------------------------------------------------------------------------
pos_window_number = 3000
neg_window_number = 3000

# -----------------------------Positive GARCH Comparision------------------------------------
pos_CHF_prediction = GARCH_pred(window_size,pos_window_number,pos_y_full_EURCHF)
joblib.dump(pos_CHF_prediction, 'GARCH3000pos_EURCHF.pkl')

pos_NOK_prediction = GARCH_pred(window_size,pos_window_number,pos_y_full_EURNOK)
joblib.dump(pos_NOK_prediction, 'GARCH3000pos_EURNOK.pkl')

pos_SEK_prediction = GARCH_pred(window_size,pos_window_number,pos_y_full_EURSEK)
joblib.dump(pos_SEK_prediction, 'GARCH3000pos_EURSEK.pkl')


# -----------------------------Negative GARCH Comparision------------------------------------
neg_CHF_prediction = GARCH_pred(window_size,neg_window_number,neg_y_full_EURCHF)
joblib.dump(neg_CHF_prediction, 'GARCH3000neg_EURCHF.pkl')

neg_NOK_prediction = GARCH_pred(window_size,neg_window_number,neg_y_full_EURNOK)
joblib.dump(neg_NOK_prediction, 'GARCH3000neg_EURNOK.pkl')

neg_SEK_prediction = GARCH_pred(window_size,neg_window_number,neg_y_full_EURSEK)
joblib.dump(neg_SEK_prediction, 'GARCH3000neg_EURSEK.pkl')
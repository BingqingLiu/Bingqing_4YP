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
    y_full1 = y_full.reshape(len(y_full),1)
    y_full = (y1_full - np.mean(y1_full))/np.std(y1_full) #normalization
    
    # y_full = np.log(y_full)
    return x_full, y_full

# -------------------------------------------------------------------------------------------
# ----------------------Data Processing-------------------------------------
# -------------------------------------------------------------------------------------------
window_size = 50 #size of each window
start_from = 0 #window starting position
vol_window = 10
# ---------------------------------------GARCH-----------------------------------------------------

# Data process
y_full_EURCHF = [LogR_EURCHF_full[i] for i in range(0, n-1) if LogR_EURCHF_full[i]!=0]
x_full_EURCHF = [i*1.0 for i in range(0, n-1) if LogR_EURCHF_full[i]!=0]
[x_EURCHF, y_EURCHF] = Normalization(x_full_EURCHF,y_full_EURCHF)

y_full_EURNOK = [LogR_EURNOK_full[i] for i in range(0, n-1) if LogR_EURNOK_full[i]!=0]
x_full_EURNOK = [i*1.0 for i in range(0, n-1) if LogR_EURNOK_full[i]!=0]
[x_EURNOK, y_EURNOK] = Normalization(x_full_EURNOK,y_full_EURNOK)

y_full_EURSEK = [LogR_EURSEK_full[i] for i in range(0, n-1) if LogR_EURSEK_full[i]!=0]
x_full_EURSEK = [i*1.0 for i in range(0, n-1) if LogR_EURSEK_full[i]!=0]
[x_EURSEK, y_EURSEK] = Normalization(x_full_EURSEK,y_full_EURSEK)


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
window_number = 3000

# -----------------------------Positive GARCH Comparision------------------------------------
CHF_prediction = GARCH_pred(window_size,window_number,y_full_EURCHF)
joblib.dump(CHF_prediction, 'GARCH3000_EURCHF.pkl')

NOK_prediction = GARCH_pred(window_size,window_number,y_full_EURNOK)
joblib.dump(NOK_prediction, 'GARCH3000_EURNOK.pkl')

SEK_prediction = GARCH_pred(window_size,window_number,y_full_EURSEK)
joblib.dump(SEK_prediction, 'GARCH3000_EURSEK.pkl')


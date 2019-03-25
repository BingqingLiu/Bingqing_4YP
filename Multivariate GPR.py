# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# #config.gpu_options.per_process_gpu_memory_fraction = 0.3
# set_session(tf.Session(config=config))
#


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
n = len(fxData.EURUSD)
LogR_EURUSD_full = [np.log(fxData.EURUSD[i+1])-np.log(fxData.EURUSD[i]) for i in range(0, n-1)]
# LogR_EURCHF_full = [np.log(fxData.EURCHF[i+1])-np.log(fxData.EURCHF[i]) for i in range(0, n-1)]
# LogR_EURNOK_full = [np.log(fxData.EURNOK[i+1])-np.log(fxData.EURNOK[i]) for i in range(0, n-1)]
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
    y_full1 = y_full.reshape(len(y_full),1)
    y_full = (y1_full - np.mean(y1_full))/np.std(y1_full) #normalization
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

# 3. Rolling Window GP regression
def predict_update(pred_range,window_size,start_from,window_number,x_asset1,x_asset2,y_asset1,y_asset2):
    total_range = window_size + pred_range
    mean_asset1 = []
    mean_asset2 = []
    var_asset1 = []
    var_asset2 = []
    lengthscale = 10

    # Rolling window input training data, set x and y input range for the model
    for i in range(0, window_number):

        x_training = x_asset2[i+start_from:window_size+i+start_from]
        x_training = np.array(x_training)
        x_training = x_training.reshape(window_size, 1)
        x_training = np.append(x_training,x_training)
        x_training = x_training.reshape(2,window_size)
        x_training = x_training.T

        y_training_asset2 = y_asset2[i+start_from:window_size+i+start_from]
        y_training_asset2 = np.array(y_training_asset2)
        y_training_asset2 = y_training_asset2.reshape(window_size,1)


        for k in range(0,len(x_asset1)):
            if x_asset1[k]>x_asset2[i+start_from]: #make sure the two x arrays are starting at roughly the same position
                index = k
                # print(k)
                break
        y_training_asset1 = y_asset1[index:window_size+index]
        y_training_asset1 = np.array(y_training_asset1)
        y_training_asset1 = y_training_asset1.reshape(window_size,1)

        y_training = [y_training_asset1, y_training_asset2]
        y_training = np.array(y_training)
        y_training = y_training.reshape(2, window_size)
        y_training = y_training.T

        # Total range
        xx_temp = x_asset2[i+start_from:total_range+i+start_from]
        xx_temp = np.array(xx_temp).reshape(total_range, 1)
        xx_temp = np.append(xx_temp,xx_temp)
        xx_temp = np.array(xx_temp).reshape(2, total_range)
        xx_temp = xx_temp.T

        # Update the model input data
        # k1 = gpflow.kernels.Matern32(1,active_dims=[0])
        # k2 = gpflow.kernels.Matern32(1, active_dims=[1])
        # kern_multi = k1+k2
        kern_multi = gpflow.kernels.Matern32(2)

        if i == 0:
            m = gpflow.models.GPR(x_training, y_training,kern=kern_multi)
            # m.kern.kernels[0].lengthscales = lengthscale
            # m.kern.lengthscales = lengthscale
            m.kern.lengthscales.trainable = False
            gpflow.train.ScipyOptimizer().minimize(m)
            m.as_pandas_table()

        m.X = x_training
        m.Y = y_training

        # optimize the hyperparameters every 50 windows
        if i % 5 == 0:
            print(i)
            m = gpflow.models.GPR(x_training, y_training, kern=kern_multi)
            # m.kern.kernels[0].lengthscales = lengthscale
            m.kern.lengthscales = lengthscale
            m.kern.lengthscales.trainable = False
            gpflow.train.ScipyOptimizer().minimize(m)
            print('after opt', m.as_pandas_table())

            # rebuild m if the function reaches its limits
            if m.kern.variance.value < 0.0001 or m.kern.variance.value > 10:
                print('rebuild m: whitenoise')
                m = gpflow.models.GPR(x_training, y_training, kern=kern_multi)
                m.kern.lengthscales = lengthscale
                m.kern.lengthscales.trainable = False
                gpflow.train.ScipyOptimizer().minimize(m)
                print(m.as_pandas_table())

            #
            # if m.kern.kernels[0].variance.value < 0.005 or m.kern.kernels[0].variance.value > 10:
            #     print('rebuild m: signal noise')
            #     m = gpflow.models.GPR(x_training, y_training, kern=gpflow.kernels.Matern32(1) + gpflow.kernels.White(1))
            #     # m.kern.kernels[0].lengthscales = lengthscale
            #     # m.kern.kernels[0].lengthscales.trainable = False
            #     gpflow.train.ScipyOptimizer().minimize(m)
            #     print(m.as_pandas_table())

        # Store GP Result
        mean_temp, var_temp = m.predict_y(xx_temp)  # return mean and var
        mean_asset1 = np.append(mean_asset1,mean_temp[:,0])
        mean_asset2 = np.append(mean_asset2,mean_temp[:,1])
        var_asset1 = np.append(var_asset1,var_temp[:,0])
        var_asset2 = np.append(var_asset2,var_temp[:,1])
    return mean_asset1, mean_asset2, var_asset1,var_asset2


# 4. Final GPR Graph Plot
def plotGP(mean_asset1,mean_asset2,var_asset1,var_asset2,window_number,window_size,pred_range,start_from,x_asset1,x_asset2,y_asset1,y_asset2):
    total_range = pred_range+window_size

    x_training = x_asset2[window_number+start_from:window_size + window_number+start_from]
    x_training = np.array(x_training)
    x_training = x_training.reshape(1,window_size)

    y_training_asset2 = y_asset2[window_number+start_from:window_size + window_number+start_from]
    y_training_asset2 = np.array(y_training_asset2)
    y_training_asset2 = y_training_asset2.reshape(1,window_size)

    for k in range(0, len(x_asset1)):
        if x_asset1[k] > x_asset2[window_number+start_from]:
            index = k
            break
    y_training_asset1= y_asset1[index:window_size + index]
    y_training_asset1 = np.array(y_training_asset1)
    y_training_asset1 = y_training_asset1.reshape(1, window_size)

    y_training = [y_training_asset1, y_training_asset2]
    y_training = np.array(y_training)
    y_training = y_training.reshape(2, window_size)

    x_test = x_asset2[window_number+window_size+start_from:total_range + window_number+start_from]
    x_test = np.array(x_test)
    x_test = x_test.reshape(1, pred_range)

    y_test_asset2 = y_asset2[window_number+window_size+start_from:total_range+window_number+start_from]
    y_test_asset2 = np.array(y_test_asset2)
    y_test_asset2 = y_test_asset2.reshape(1, pred_range)

    y_test_asset1 = y_asset1[index+window_size:total_range+index]
    y_test_asset1 = np.array(y_test_asset1)
    y_test_asset1 = y_test_asset1.reshape(1, pred_range)

    y_test = [y_test_asset1, y_test_asset2]
    y_test = np.array(y_test)
    y_test = y_test.reshape(2, pred_range)

    xx_window = x_asset2[window_number+ start_from:total_range + window_number + start_from]
    xx_window = np.array(xx_window).reshape(len(xx_window), 1)

    plt.plot(x_training[0], y_training[0], 'bx', mew=2)  # EURUSD
    plt.plot(x_test[0], y_test[0], 'bx', mew=2)
    plt.plot(x_training[0], y_training[1], 'gx', mew=2)  # EURCHF
    plt.plot(x_test[0], y_test[1], 'gx', mew=2)

    plt.plot(xx_window[:, 0], mean_asset1[window_number], 'C0', lw=2)
    plt.plot(xx_window[:, 0], mean_asset2[window_number], 'C2', lw=2)
    plt.fill_between(xx_window[:, 0],
                     mean_asset1[window_number] - np.sqrt(var_asset1[window_number]),
                     # how to determine coefficient 0.001?
                     mean_asset1[window_number] + np.sqrt(var_asset1[window_number]),
                     color='C0', alpha=0.2)
    plt.fill_between(xx_window[:, 0],
                     mean_asset2[window_number] - np.sqrt(var_asset2[window_number]),
                     # how to determine coefficient 0.001?
                     mean_asset2[window_number] + np.sqrt(var_asset2[window_number]),
                     color='C2', alpha=0.2)
    plt.title('GPR graph of window %d' %(window_number))
    plt.show()
# -------------------------------------------------------------------------------------------
# ----------------------section 2 Implementation-----------------------------
# -------------------------------------------------------------------------------------------
pred_range = 10 #predicted range
window_size = 50 #size of each window
start_from = 0 #window starting position
vol_window = 10

# # ---------------------------------------Positive GPR-----------------------------------------------------
#
# Data process

# # 1. EURUSD
# Positive_y_EURUSD = [LogR_EURUSD_full[i] for i in range(0, n-1) if LogR_EURUSD_full[i]>0]
# Positive_x_EURUSD = [i*1.0 for i in range(0, n-1) if LogR_EURUSD_full[i]>0]
# [pos_x_EURUSD, pos_y_EURUSD] = Normalization(Positive_x_EURUSD,Positive_y_EURUSD)
#
# # # 2. EURCHF
# # Positive_y_EURCHF = [LogR_EURCHF_full[i] for i in range(0, n-1) if LogR_EURCHF_full[i]>0]
# # Positive_x_EURCHF = [i*1.0 for i in range(0, n-1) if LogR_EURCHF_full[i]>0]
# # [pos_x_EURCHF, pos_y_EURCHF] = Normalization(Positive_x_EURCHF,Positive_y_EURCHF)
#
# # # # 3. EURNOK
# # Positive_y_EURNOK = [LogR_EURNOK_full[i] for i in range(0, n-1) if LogR_EURNOK_full[i]>0]
# # Positive_x_EURNOK = [i*1.0 for i in range(0, n-1) if LogR_EURNOK_full[i]>0]
# # [pos_x_EURNOK, pos_y_EURNOK] = Normalization(Positive_x_EURNOK,Positive_y_EURNOK)
#
# # 4. EURSEK
# Positive_y_EURSEK = [LogR_EURSEK_full[i] for i in range(0, n-1) if LogR_EURSEK_full[i]>0]
# Positive_x_EURSEK = [i*1.0 for i in range(0, n-1) if LogR_EURSEK_full[i]>0]
# [pos_x_EURSEK, pos_y_EURSEK] = Normalization(Positive_x_EURSEK,Positive_y_EURSEK)
#
# # 3. Volatility
# # pos_vol_EURCHF = Vol(pos_y_EURCHF,vol_window)
# pos_vol_EURUSD = Vol(pos_y_EURUSD,vol_window)
# # pos_vol_EURNOK = Vol(pos_y_EURNOK,vol_window)
# pos_vol_EURSEK = Vol(pos_y_EURSEK,vol_window)
#
# # Gaussian Process:
# # 1. Parameter Initialization
# pos_window_number = 1000#int(len(x_full)-window_size) #number of windows
#
# # 2. Rolling Window GPR
# [mean_pos_EURUSD, mean_pos_EURSEK, var_pos_EURUSD,var_pos_EURSEK] = predict_update(pred_range,window_size,start_from,pos_window_number,pos_x_EURUSD,pos_x_EURSEK,pos_vol_EURUSD,pos_vol_EURSEK)
# # print(mean_pos_EURUSD,np.shape(mean_pos_EURUSD),'mean_pos_EURUSD')
# mean_pos_EURUSD = np.array_split(mean_pos_EURUSD, pos_window_number)
# mean_pos_EURSEK = np.array_split(mean_pos_EURSEK, pos_window_number)
# var_pos_EURUSD = np.array_split(var_pos_EURUSD, pos_window_number)
# var_pos_EURSEK = np.array_split(var_pos_EURSEK, pos_window_number)
#
#
#
# # 3. Save data
# # joblib.dump(mean_pos_EURUSD, 'mean_pos_EURUSD_multi3000_fixed.pkl')
# joblib.dump(mean_pos_EURSEK, 'mean_pos_EURSEK_multi2-3000_fixed.pkl')
# # joblib.dump(var_pos_EURUSD, 'var_pos_EURUSD_multi3000_fixed.pkl')
# joblib.dump(var_pos_EURSEK, 'var_pos_EURSEK_multi2-3000_fixed.pkl')
#
# # 2.1 Plots
# plotGP(mean_pos_EURUSD,mean_pos_EURSEK,var_pos_EURUSD,var_pos_EURSEK,10,window_size,pred_range,start_from,pos_x_EURUSD,pos_x_EURSEK,pos_vol_EURUSD,pos_vol_EURSEK)
#

# # ---------------------------------------Negative GPR-----------------------------------------------------
#
# #
# Data process

# 1. EURUSD
Negative_y_EURUSD = [-LogR_EURUSD_full[i] for i in range(0, n-1) if LogR_EURUSD_full[i]<0]
Negative_x_EURUSD = [i*1.0 for i in range(0, n-1) if LogR_EURUSD_full[i]<0]
[neg_x_EURUSD, neg_y_EURUSD] = Normalization(Negative_x_EURUSD,Negative_y_EURUSD)

# # 2. EURCHF
# Negative_y_EURCHF = [-LogR_EURCHF_full[i] for i in range(0, n-1) if LogR_EURCHF_full[i]<0]
# Negative_x_EURCHF = [i*1.0 for i in range(0, n-1) if LogR_EURCHF_full[i]<0]
# [neg_x_EURCHF, neg_y_EURCHF] = Normalization(Negative_x_EURCHF,Negative_y_EURCHF)

# # 3. EURNOK
# Negative_y_EURNOK = [-LogR_EURNOK_full[i] for i in range(0, n-1) if LogR_EURNOK_full[i]<0]
# Negative_x_EURNOK = [i*1.0 for i in range(0, n-1) if LogR_EURNOK_full[i]<0]
# [neg_x_EURNOK, neg_y_EURNOK] = Normalization(Negative_x_EURNOK,Negative_y_EURNOK)

# 4. EURSEK
Negative_y_EURSEK = [-LogR_EURSEK_full[i] for i in range(0, n-1) if LogR_EURSEK_full[i]<0]
Negative_x_EURSEK = [i*1.0 for i in range(0, n-1) if LogR_EURSEK_full[i]<0]
[neg_x_EURSEK, neg_y_EURSEK] = Normalization(Negative_x_EURSEK,Negative_y_EURSEK)


# 3. Volatility
# neg_vol_EURCHF = Vol(neg_y_EURCHF,vol_window)
neg_vol_EURUSD = Vol(neg_y_EURUSD,vol_window)
# neg_vol_EURNOK = Vol(neg_y_EURNOK,vol_window)
neg_vol_EURSEK = Vol(neg_y_EURSEK,vol_window)

# Gaussian Process:
# 1. Parameter Initialization
neg_window_number = 3000#int(len(x_full)-window_size) #number of windows


# 2. Rolling Window GPR

#
# [mean_neg_EURUSD, mean_neg_EURNOK, var_neg_EURUSD,var_neg_EURNOK] = predict_update(pred_range,window_size,start_from,neg_window_number,neg_x_EURUSD,neg_x_EURNOK,neg_vol_EURUSD,neg_vol_EURNOK)
# mean_neg_EURUSD = np.array_split(mean_neg_EURUSD, neg_window_number)
# mean_neg_EURNOK = np.array_split(mean_neg_EURNOK, neg_window_number)
# var_neg_EURUSD = np.array_split(var_neg_EURUSD, neg_window_number)
# var_neg_EURNOK = np.array_split(var_neg_EURNOK, neg_window_number)


[mean_neg_EURUSD, mean_neg_EURSEK, var_neg_EURUSD,var_neg_EURSEK] = predict_update(pred_range,window_size,start_from,neg_window_number,neg_x_EURUSD,neg_x_EURSEK,neg_vol_EURUSD,neg_vol_EURSEK)
mean_neg_EURUSD = np.array_split(mean_neg_EURUSD, neg_window_number)
mean_neg_EURSEK = np.array_split(mean_neg_EURSEK, neg_window_number)
var_neg_EURUSD = np.array_split(var_neg_EURUSD, neg_window_number)
var_neg_EURSEK = np.array_split(var_neg_EURSEK, neg_window_number)


#3. Save data
joblib.dump(mean_neg_EURUSD, 'mean_neg_EURUSDSEK_multi3000_fixed.pkl')
joblib.dump(mean_neg_EURSEK, 'mean_neg_EURSEK_multi3000_fixed.pkl')
joblib.dump(var_neg_EURUSD, 'var_neg_EURUSDSEK_multi3000_fixed.pkl')
joblib.dump(var_neg_EURSEK, 'var_neg_EURSEK_multi3000_fixed.pkl')

# #3. Save data
# # joblib.dump(mean_neg_EURUSD, 'mean_neg_EURUSDNOK_multi2-3000_fixed.pkl')
# joblib.dump(mean_neg_EURSEK, 'mean_neg_EURSEK_multi2000_fixed.pkl')
# # joblib.dump(var_neg_EURUSD, 'var_neg_EURUSDSEK_multi2000_fixed.pkl')
# joblib.dump(var_neg_EURSEK, 'var_neg_EURSEK_multi2000_fixed.pkl')


# 2.1 Plots
plotGP(mean_neg_EURUSD,mean_neg_EURSEK,var_neg_EURUSD,var_neg_EURSEK,999,window_size,pred_range,start_from,neg_x_EURUSD,neg_x_EURSEK,neg_vol_EURUSD,neg_vol_EURSEK)


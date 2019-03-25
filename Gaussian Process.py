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
LogR_EURCHF_full = [np.log(fxData.EURCHF[i+1])-np.log(fxData.EURCHF[i]) for i in range(0, n-1)]
# LogR_EURNOK_full = [np.log(fxData.EURNOK[i+1])-np.log(fxData.EURNOK[i]) for i in range(0, n-1)]
# LogR_EURSEK_full = [np.log(fxData.EURSEK[i+1])-np.log(fxData.EURSEK[i]) for i in range(0, n-1)]

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

# 3. Initial Hyperparameters Optimization
def GPInitial(start_from,window_size,pre_range,x_full,y_full):
    lengthscale = 5
    x_training = x_full[0:window_size];  # first 100 data as training data
    x_training = x_training.reshape(window_size,1)
    y_training = y_full[0:window_size];
    y_training = y_training.reshape(window_size, 1)
    x_test = x_full[window_size: pre_range + window_size];  # next 50 data as test data
    y_test = y_full[window_size: pre_range + window_size];

    m_full = gpflow.models.GPR(x_training, y_training, kern=gpflow.kernels.Matern32(1)) #+ gpflow.kernels.White(1))
    m_full.kern.lengthscales = lengthscale
    m_full.kern.lengthscales.trainable = False
    gpflow.train.ScipyOptimizer().minimize(m_full)
    m_full.as_pandas_table()
    return m_full


# 4. Rolling Window GP regression
def predict_full_update(m,pred_range,window_size,start_from,window_number,x_full,y_full):
    total_range = window_size + pred_range
    mean_full = []
    var_full = []
    x_training = []
    y_training = []
    x_test = []
    y_test = []
    lengthscale = 5

    # Rolling window input training data, set x and y input range for the model
    for i in range(0, window_number):
        x_training = x_full[i + start_from:window_size + i + start_from]; #first 50 data
        x_training = x_training.reshape(window_size, 1)
        y_training = y_full[i + start_from:window_size + i + start_from];
        y_training = y_training.reshape(window_size, 1)

        # Total range
        x_total = x_full[i + start_from:total_range + i + start_from]
        xx_temp = np.array(x_total).reshape(total_range, 1)

        # Update the model input data
        m.X = x_training
        m.Y = y_training

        # set hyperparameters as previously trained/optimized ones
        #         if i>0:
        #             m.assign(val)

        # optimize the hyperparameters every 50 windows
        if i % 5 == 0:
            print(i)
            m.kern.lengthscales = lengthscale
            m.kern.lengthscales.trainable = False
            gpflow.train.ScipyOptimizer().minimize(m)
            print('after opt', m.as_pandas_table())
            # val = m.read_values() #to get the parameters

            # # rebuild m if the function reaches its limits
            # if m.kern.lengthscales.value < 0.2 or m.kern.lengthscales.value > 45:
            #     print('rebuild m: lengthscale')
            #     m = gpflow.models.GPR(x_training, y_training, kern=gpflow.kernels.Matern32(1) + gpflow.kernels.White(1))
            #     m.kern.lengthscales = lengthscale
            #     m.kern.lengthscales.trainable = False
            #     gpflow.train.ScipyOptimizer().minimize(m)
            #     print(m.as_pandas_table())
            #

            if m.kern.variance.value < 0.0001 or m.kern.variance.value > 10:
                print('rebuild m: signal noise')
                m = gpflow.models.GPR(x_training, y_training, kern=gpflow.kernels.Matern32(1))
                m.kern.lengthscales = lengthscale
                m.kern.lengthscales.trainable = False
                gpflow.train.ScipyOptimizer().minimize(m)
                print(m.as_pandas_table())
                # val = m.read_values()

            # Store parameters
            # parameters_new = pd.DataFrame(np.array([[i,m.kern.kernels[0].lengthscales.value,m.kern.kernels[0].variance.value,m.kern.kernels[1].variance.value]]),columns=['i', 'lengscale', 'signal var', 'noise var'])
            # parameters = parameters.append(parameters_new)
            # print(parameters,'parameters1')
        # Store GP Result
        mean_temp, var_temp = m.predict_y(xx_temp)  # return mean and var
        mean_temp = mean_temp.tolist()
        var_temp = var_temp.tolist()
        mean_full = np.append(mean_full, mean_temp)
        var_full = np.append(var_full, var_temp)
        # print(mean_full)
    mean_full = np.array(mean_full)
    var_full = np.array(var_full)
    # print(parameters,'parameters final')
    return mean_full, var_full  #,parameters

# 5. Final GPR Graph Plot
def plotGP(mean,var,window_number,window_size,pred_range,x_full,y_full):
    total_range = pred_range+window_size
    mean_window = mean[window_number]
    var_window = var[window_number]

    x_train = x_full[window_number: window_number+window_size]; #first 100 data as training data
    y_train = y_full[window_number: window_number+window_size];

    x_test = x_full[window_number+window_size: window_number+total_range]; #next 50 data as test data
    y_test = y_full[window_number+window_size: window_number+total_range];

    x_total = x_full[window_number:total_range+window_number];
    xx_window = np.array(x_total).reshape(total_range, 1)

    plt.plot(x_train, y_train, 'kx', mew=2)
    plt.plot(x_test, y_test, 'rx', mew=2)
    plt.plot(xx_window, mean_window, 'C0', lw=2)
    plt.fill_between(xx_window[:,0], mean_window - np.sqrt(var_window), mean_window + np.sqrt(var_window),color='C0', alpha=0.2)
    plt.title('GPR graph of window %d' %(window_number))
    plt.show()


# -------------------------------------------------------------------------------------------
# ----------------------section 2 Implementation-----------------------------
# -------------------------------------------------------------------------------------------
pred_range = 10 #predicted range
window_size = 50 #size of each window
start_from = 0 #window starting position
vol_window = 10
# ---------------------------------------Positive GPR-----------------------------------------------------
#
# Data process
# Positive_y_full = [LogR_EURCHF_full[i] for i in range(0, n-1) if LogR_EURCHF_full[i]>0]
# Positive_x_full = [i*1.0 for i in range(0, n-1) if LogR_EURCHF_full[i]>0]
# Positive_y_full = [LogR_EURNOK_full[i] for i in range(0, n-1) if LogR_EURNOK_full[i]>0]
# Positive_x_full = [i*1.0 for i in range(0, n-1) if LogR_EURNOK_full[i]>0]
Positive_y_full = [LogR_EURCHF_full[i] for i in range(0, n-1) if LogR_EURCHF_full[i]>0]
Positive_x_full = [i*1.0 for i in range(0, n-1) if LogR_EURCHF_full[i]>0]
[pos_x_full, pos_y_full] = Normalization(Positive_x_full,Positive_y_full )
pos_y_vol = Vol(pos_y_full,vol_window)
plt.plot(pos_x_full,pos_y_vol)
plt.show()
# Gaussian Process:
# 1. Parameter Initialization
pos_window_number = 3000#int(len(x_full)-window_size) #number of windows

m_pos_full = GPInitial(start_from,window_size,pred_range,pos_x_full,pos_y_vol)

# 2 Rolling Window GPR

[pos_mean_full, pos_var_full] = predict_full_update(m_pos_full,pred_range,window_size,start_from,pos_window_number,pos_x_full,pos_y_vol)
pos_mean_full = np.array_split(pos_mean_full, pos_window_number)
pos_var_full = np.array_split(pos_var_full, pos_window_number)
# 2 Load computed data
joblib.dump(pos_mean_full, 'mean_CHF_3000_pos_fixed5.pkl')
joblib.dump(pos_var_full, 'var_CHF_3000_pos_fixed5.pkl')


# # 2.1 Plots
plotGP(pos_mean_full ,pos_var_full,10,window_size,pred_range,pos_x_full, pos_y_vol)


# # ---------------------------------------Negative GPR-----------------------------------------------------
# Data process
# Negative_y_full = [-LogR_EURCHF_full[i] for i in range(0, n-1) if LogR_EURCHF_full[i]<0]
# Negative_x_full = [i*1.0 for i in range(0, n-1) if LogR_EURCHF_full[i]<0]

# Negative_y_full = [-LogR_EURNOK_full[i] for i in range(0, n-1) if LogR_EURNOK_full[i]<0]
# Negative_x_full = [i*1.0 for i in range(0, n-1) if LogR_EURNOK_full[i]<0]
#
Negative_y_full = [-LogR_EURCHF_full[i] for i in range(0, n-1) if LogR_EURCHF_full[i]<0]
Negative_x_full = [i*1.0 for i in range(0, n-1) if LogR_EURCHF_full[i]<0]

[neg_x_full, neg_y_full] = Normalization(Negative_x_full,Negative_y_full )
neg_y_vol = Vol(neg_y_full,vol_window)


# Gaussian Process:
# 1. Parameter Initialization
neg_window_number = 3000 #int(len(x_full)-window_size) #number of windows

m_neg_full = GPInitial(start_from,window_size,pred_range,neg_x_full,neg_y_vol)


# 2 Rolling Window GPR

[neg_mean_full, neg_var_full] = predict_full_update(m_neg_full,pred_range,window_size,start_from,neg_window_number,neg_x_full,neg_y_vol)
neg_mean_full = np.array_split(neg_mean_full, neg_window_number)
neg_var_full = np.array_split(neg_var_full, neg_window_number)

# 2 Load computed data

joblib.dump(neg_mean_full, 'mean_CHF_3000_neg_fixed5.pkl')
joblib.dump(neg_var_full, 'var_CHF_3000_neg_fixed5.pkl')

# neg_mean_load =  joblib.load('neg_mean_CHF_5000.pkl')
# neg_var_load =  joblib.load('neg_var_CHF_5000.pkl')

# 2.1 Plots
plotGP(neg_mean_full,neg_var_full,500,window_size,pred_range,neg_x_full, neg_y_vol)
#

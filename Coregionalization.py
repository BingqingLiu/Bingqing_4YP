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

# 2. Coregionalization
def Coregionalization(pred_range,window_size,window_number,start_from,x_asset1,y_asset1,x_asset2,y_asset2):
    # Parameters initialization
    total_range = window_size + pred_range
    x_training = []
    y_training = []
    x_test = []
    y_test = []
    mean_asset1 = []
    var_asset1 = []
    mean_asset2 = []
    var_asset2 = []

    lengthscale = 10

    # Rolling window input training data, set x and y input range for the model
    for i in range(0, window_number):
        asset2_training_x = x_asset2[i+start_from:window_size + i+start_from];
        asset2_training_x = np.array(asset2_training_x).reshape(len(asset2_training_x), 1)

        asset2_training_y = y_asset2[i+start_from:window_size + i+start_from];
        asset2_training_y = np.array(asset2_training_y).reshape(len(asset2_training_y), 1)

        for k in range(0, len(x_asset1)):
            if x_asset1[k] > x_asset2[i+start_from]:
                index = k
                break

        asset1_training_x = x_asset1[index:window_size + index]
        asset1_training_x = np.array(asset1_training_x)
        asset1_training_x = asset1_training_x.reshape(window_size, 1)

        asset1_training_y = y_asset1[index:window_size+index]
        asset1_training_y = np.array(asset1_training_y)
        asset1_training_y = asset1_training_y.reshape(window_size,1)

        # Total range
        xx_temp = x_asset2[i+start_from:total_range+i+start_from]
        xx_temp = np.array(xx_temp).reshape(len(xx_temp), 1)


        # optimize the hyperparameters every 50 windows
        if i % 5 == 0:
            print(i)

            # a Coregionalization kernel. The base kernel is Matern, and acts on the first ([0]) data dimension.
            # the 'Coregion' kernel indexes the outputs, and actos on the second ([1]) data dimension
            k1 = gpflow.kernels.Matern32(1, active_dims=[0])

            coreg = gpflow.kernels.Coregion(1, output_dim=2, rank=1, active_dims=[1])
            kern = k1 * coreg

            # build a variational model. This likelihood switches between Student-T noise with different variances:
            lik = gpflow.likelihoods.SwitchedLikelihood([gpflow.likelihoods.StudentT(), gpflow.likelihoods.StudentT()])

            # Augment the time data with ones or zeros to indicate the required output dimension
            X_augmented = np.vstack((np.hstack((asset1_training_x, np.zeros_like(asset1_training_x))),
                                     np.hstack((asset2_training_x, np.ones_like(asset2_training_x)))))

            # Augment the Y data to indicate which likeloihood we should use
            Y_augmented = np.vstack((np.hstack((asset1_training_y, np.zeros_like(asset1_training_x))),
                                     np.hstack((asset2_training_y, np.ones_like(asset2_training_x)))))

            # now buld the GP model as normal
            m = gpflow.models.VGP(X_augmented, Y_augmented, kern=kern, likelihood=lik, num_latent=1)
            m.kern.kernels[0].lengthscales = lengthscale
            m.kern.kernels[0].lengthscales.trainable = False

            # fit the covariance function parameters
            # m.kern.coregion.W = np.random.randn(2, 1)
            gpflow.train.ScipyOptimizer().minimize(m)
            print(m.as_pandas_table())

        mean_temp_asset1, var_temp_asset1 = m.predict_f(np.hstack((xx_temp, np.zeros_like(xx_temp))))
        mean_temp_asset2, var_temp_asset2 = m.predict_f(np.hstack((xx_temp, np.ones_like(xx_temp))))

        mean_temp_asset1 = mean_temp_asset1.tolist()
        var_temp_asset1 = var_temp_asset1.tolist()
        mean_temp_asset2 = mean_temp_asset2.tolist()
        var_temp_asset2 = var_temp_asset2.tolist()

        mean_asset1 = np.append(mean_asset1, mean_temp_asset1)
        var_asset1 = np.append(var_asset1, var_temp_asset1)
        mean_asset2 = np.append(mean_asset2, mean_temp_asset2)
        var_asset2 = np.append(var_asset2, var_temp_asset2)
        print(mean_asset1,np.shape(mean_asset1),'step1')
    mean_asset1 = np.array(mean_asset1)
    var_asset1 = np.array(var_asset1)
    mean_asset2 = np.array(mean_asset2)
    var_asset2 = np.array(var_asset2)

    print(mean_asset1,np.shape(mean_asset1),'step2')
    return mean_asset1, var_asset1, mean_asset2, var_asset2


# 3. Coregionalization GPR Graph Plot
def Coregional_Graph(window_number,window_size,pred_range,start_from,mean_asset1,mean_asset2,var_asset1,var_asset2,x_asset1,x_asset2,y_asset1,y_asset2):
    total_range = pred_range + window_size

    mean_asset1 = mean_asset1[window_number]
    mean_asset2 = mean_asset2[window_number]
    var_asset1 = var_asset1[window_number]
    var_asset2 = var_asset2[window_number]

    x_train_asset2 = x_asset2[window_number +start_from: window_number +start_from + window_size];  # first 100 data as training data
    y_train_asset2 = y_asset2[window_number +start_from: window_number +start_from + window_size];
    x_test_asset2 = x_asset2[window_number +start_from + window_size: window_number +start_from + total_range]
    y_test_asset2 = y_asset2[window_number +start_from + window_size: window_number +start_from + total_range]

    for k in range(0, len(x_asset1)):
        if x_asset1[k] > x_asset2[window_number+start_from]:
            index = k
            break

    x_train_asset1 = x_asset1[index:window_size + index];  # first 100 data as training data
    y_train_asset1 = y_asset1[index:window_size + index];
    x_test_asset1 = x_asset1[index+window_size:total_range+index];
    y_test_asset1 = y_asset1[index+window_size:total_range+index];

    xx_window = x_asset2[window_number+ start_from:total_range + window_number + start_from];
    xx_window = np.array(xx_window).reshape(len(xx_window), 1)

    plt.plot(x_train_asset1, y_train_asset1, 'bx', mew=2)  # EURUSD
    plt.plot(x_test_asset1, y_test_asset1, 'bx', mew=2)
    plt.plot(x_train_asset1, y_train_asset2, 'gx', mew=2)  # EURSEK #!!! SHOULD BE X_ASSET2!!! CHECK RANGE
    plt.plot(x_test_asset1, y_test_asset2, 'gx', mew=2) #!!! SHOULD BE X_ASSET2!!! CHECK RANGE

    plt.plot(xx_window, mean_asset1, 'C0', lw=2)
    plt.plot(xx_window, mean_asset2, 'C2', lw=2)
    plt.fill_between(xx_window[:, 0],
                     mean_asset1 - np.sqrt(var_asset1),  # how to determine coefficient 0.001?
                     mean_asset1 + np.sqrt(var_asset1),
                     color='C0', alpha=0.2)
    plt.fill_between(xx_window[:, 0],
                     mean_asset2 - np.sqrt(var_asset2),  # how to determine coefficient 0.001?
                     mean_asset2 + np.sqrt(var_asset2),
                     color='C2', alpha=0.2)

    plt.title('Coregionalization GP graph of window %d' %(window_number))
    plt.show()



# -------------------------------------------------------------------------------------------
# ----------------------section 2 Implementation-----------------------------
# -------------------------------------------------------------------------------------------
pred_range = 10 #predicted range
window_size = 50 #size of each window
start_from = 1500
vol_window = 10
# ---------------------------------------Positive GPR-----------------------------------------------------

#
# Data process
#
# 1. EURUSD
Positive_y_EURUSD = [LogR_EURUSD_full[i] for i in range(0, n-1) if LogR_EURUSD_full[i]>0]
Positive_x_EURUSD = [i*1.0 for i in range(0, n-1) if LogR_EURUSD_full[i]>0]
[pos_x_EURUSD, pos_y_EURUSD] = Normalization(Positive_x_EURUSD,Positive_y_EURUSD)

# # 2. EURCHF
# Positive_y_EURCHF = [LogR_EURCHF_full[i] for i in range(0, n-1) if LogR_EURCHF_full[i]>0]
# Positive_x_EURCHF = [i*1.0 for i in range(0, n-1) if LogR_EURCHF_full[i]>0]
# [pos_x_EURCHF, pos_y_EURCHF] = Normalization(Positive_x_EURCHF,Positive_y_EURCHF)

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
# Coregionalzation Gaussian Process:

# 1. Parameter Initialization
pos_window_number = 1500#int(len(x_full)-window_size) #number of windows

# 2. Coregionalzation GPR

[mean_pos_EURUSD, var_pos_EURUSD, mean_pos_EURSEK, var_pos_EURSEK] = Coregionalization(pred_range,window_size,pos_window_number,start_from,pos_x_EURUSD,pos_vol_EURUSD,pos_x_EURSEK,pos_vol_EURSEK)

mean_pos_EURUSD = np.array_split(mean_pos_EURUSD, pos_window_number)
mean_pos_EURSEK = np.array_split(mean_pos_EURSEK, pos_window_number)
var_pos_EURUSD = np.array_split(var_pos_EURUSD, pos_window_number)
var_pos_EURSEK = np.array_split(var_pos_EURSEK, pos_window_number)

# 3. Save data
joblib.dump(mean_pos_EURUSD, 'mean_pos_EURUSDSEK_coregion_15-3000.pkl')
joblib.dump(mean_pos_EURSEK, 'mean_pos_EURSEK_coregion_15-3000.pkl')
joblib.dump(var_pos_EURUSD, 'var_pos_EURUSDSEK_coregion_15-3000.pkl')
joblib.dump(var_pos_EURSEK, 'var_pos_EURSEK_coregion_15-3000.pkl')

# 4. Coregionalzation Plot
Coregional_Graph(999,window_size,pred_range,start_from,mean_pos_EURUSD, mean_pos_EURSEK, var_pos_EURUSD,var_pos_EURSEK,pos_x_EURUSD,pos_x_EURSEK,pos_vol_EURUSD,pos_vol_EURSEK)

# # ---------------------------------------Negative GPR-----------------------------------------------------
# # Data process
#
# # 1. EURUSD
# Negative_y_EURUSD = [-LogR_EURUSD_full[i] for i in range(0, n-1) if LogR_EURUSD_full[i]<0]
# Negative_x_EURUSD = [i*1.0 for i in range(0, n-1) if LogR_EURUSD_full[i]<0]
# [neg_x_EURUSD, neg_y_EURUSD] = Normalization(Negative_x_EURUSD,Negative_y_EURUSD)
#
# # # 2. EURCHF
# # Negative_y_EURCHF = [-LogR_EURCHF_full[i] for i in range(0, n-1) if LogR_EURCHF_full[i]<0]
# # Negative_x_EURCHF = [i*1.0 for i in range(0, n-1) if LogR_EURCHF_full[i]<0]
# # [neg_x_EURCHF, neg_y_EURCHF] = Normalization(Negative_x_EURCHF,Negative_y_EURCHF)
#
# # # 2. EURNOK
# # Negative_y_EURNOK = [-LogR_EURNOK_full[i] for i in range(0, n-1) if LogR_EURNOK_full[i]<0]
# # Negative_x_EURNOK = [i*1.0 for i in range(0, n-1) if LogR_EURNOK_full[i]<0]
# # [neg_x_EURNOK, neg_y_EURNOK] = Normalization(Negative_x_EURNOK,Negative_y_EURNOK)
#
# # 2. EURSEK
# Negative_y_EURSEK = [-LogR_EURSEK_full[i] for i in range(0, n-1) if LogR_EURSEK_full[i]<0]
# Negative_x_EURSEK = [i*1.0 for i in range(0, n-1) if LogR_EURSEK_full[i]<0]
# [neg_x_EURSEK, neg_y_EURSEK] = Normalization(Negative_x_EURSEK,Negative_y_EURSEK)
# #
#
# # # 3. Volatility
# # neg_vol_EURNOK = Vol(neg_y_EURNOK,vol_window)
# neg_vol_EURUSD = Vol(neg_y_EURUSD,vol_window)
# neg_vol_EURSEK = Vol(neg_y_EURSEK,vol_window)
#
# # 1. Parameter Initialization
# neg_window_number = 3000#int(len(x_full)-window_size) #number of windows
# # print(np.shape(neg_vol_EURCHF),np.shape(neg_vol_EURUSD))
#
# #  2. Coregionalzation GPR
# [mean_neg_EURUSD, var_neg_EURUSD, mean_neg_EURSEK, var_neg_EURSEK] = Coregionalization(pred_range,window_size,neg_window_number,start_from,neg_x_EURUSD,neg_vol_EURUSD,neg_x_EURSEK,neg_vol_EURSEK)
#
# mean_neg_EURUSD = np.array_split(mean_neg_EURUSD, neg_window_number)
# mean_neg_EURSEK = np.array_split(mean_neg_EURSEK, neg_window_number)
# var_neg_EURUSD = np.array_split(var_neg_EURUSD, neg_window_number)
# var_neg_EURSEK = np.array_split(var_neg_EURSEK, neg_window_number)
#
# # 3. Save data
# joblib.dump(mean_neg_EURUSD, 'mean_neg_EURSEKUSD_coregion_3000.pkl')
# joblib.dump(mean_neg_EURSEK, 'mean_neg_EURSEK_coregion_3000.pkl')
# joblib.dump(var_neg_EURUSD, 'var_neg_EURSEKUSD_coregion_3000.pkl')
# joblib.dump(var_neg_EURSEK, 'var_neg_EURSEK_coregion_3000.pkl')
#
# # 4. Coregionalzation Plot
# Coregional_Graph(990,window_size,pred_range,start_from,mean_neg_EURUSD, mean_neg_EURSEK, var_neg_EURUSD,var_neg_EURSEK,neg_x_EURUSD,neg_x_EURSEK,neg_vol_EURUSD,neg_vol_EURSEK)
#

#%reset
import numpy as np
from statsmodels.tsa.arima_process import arma_generate_sample

import matplotlib.pyplot as plt

def ar_gen(n, ar):
    np.random.seed(12345)
    maparams = np.array([0,0])
    ar = np.r_[1, -ar] # add zero-lag and negate
    ma = np.r_[1, maparams] # add zero-lag
    y = arma_generate_sample(ar, ma, n)
    return y, ar

def predict(y,ar):
    yt=y.copy()
    for i in range(len(y)-len(ar)):
        yt[i+2]=ar[0]*y[i+1]+ar[1]*y[i]
    
    SSE=np.sum((y-yt)**2)/len(yt)
    print(SSE)
    return y,yt, SSE

def predict_yt_y_t(y,yt,ar):
    y_t=y.copy()
    for i in range(len(y)-len(ar)):
        y_t[i+2]=ar[0]*yt[i+1]+ar[1]*yt[i]
    
    SSE=np.sum((y-y_t)**2)/len(yt)
    print(SSE)
    return y,yt, y_t, SSE

def predict_arg(y,ar):
    yt=y.copy()
    for i in range(len(y)-len(ar)):
        yt[i+len(ar)]=0
        for j in range(len(ar)):
#            yt[i+len(ar)]=ar[0]*y[i+1]+ar[1]*y[i]
            yt[i+len(ar)]=ar[j]*y[i-j+1]+yt[i+len(ar)]
    
    SSE=np.sum((y-yt)**2)/len(yt)
    print(SSE)
    return y,yt, SSE

def plotting(y,yt):
    x=np.arange(len(y))
    plt.plot(x[100:199], y[100:199], color='black', label='$Origin$')
    plt.plot(x[100:199], yt[100:199], color='red', label='$Predicted$')
    plt.legend(loc="lower right")

def plotting3(y,yt,y_t):
    x=np.arange(len(y))
    plt.plot(x[4001:4100], y[4001:4100], color='black', label='$Origin$')
    plt.plot(x[4001:4100], yt[4001:4100], color='red', label='$Predicted$')
    plt.plot(x[4001:4100], y_t[4001:4100], color='blue', label='$Two-Step Predicted$')

    plt.legend(loc="lower right")

def normal(y):
    x1=(y-np.min(y))/(np.max(y)-np.min(y))
    return x1 

def re_normal(y1,y):
    y2=y1*(np.max(y)-np.min(y))+np.min(y)
    return y2 

def re_normal(x,y):
    x=x*(max(y)-min(y))+min(y)
    return x

# Time Series Prediction AR(2)
ar=np.array([.75, -.25])
y, ar=ar_gen(10000, ar)
y,yt,SSE=predict(y,ar)
plotting(y,yt)
 
# Time Series Prediction AR(4)
ar=np.array([-1.876,-1.781,-1.201, -0.373])
y, ar=ar_gen(10000, ar)
y,yt,SSE=predict_arg(y,ar)
plotting(y,yt)
   
# Time Series AR(2)
ar=np.array([1.49,-0.653])
y, ar=ar_gen(10000, ar)
y,yt,SSE=predict_arg(y,ar)
#plotting(y,yt)
y,yt,y_t,SSE=predict_yt_y_t(y,yt,ar)
plotting3(y,yt,y_t)

# Normalization Prediction
ar=np.array([.75, -.25])
y, ar=ar_gen(1000, ar)
yn=normal(y)
y,yt,SSE=predict(yn,ar)
plotting(y,yt)

# Normalization Prediction
ar=np.array([1.49,-0.653])
y, ar=ar_gen(1000, ar)
yn=normal(y)
yn,yt,SSE=predict(yn, ar)
plotting(yn,yt)

# Re-Normalization Prediction

yrn=re_normal(yt,y)
plotting(y, yrn)









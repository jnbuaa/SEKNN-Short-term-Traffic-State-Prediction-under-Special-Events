# -*- coding: utf-8 -*-
"""
@author: Nan Ji
"""
import numpy as np
import pandas as pd
import glob
import time
import warnings
from preprocess import DataPreprocess
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

warnings.filterwarnings('ignore')

dp = DataPreprocess(inputstep=6,predstep=4)
inputstep = dp.inputstep
predstep = dp.predstep
maxv = dp.maxv
link = 257

print('loading and partitioning dataset...')
hplist = glob.glob(r'E:/gongtiS/Data/NormalTarget/*csv')[30:47] #July 01-16
#vplist = glob.glob(r'E:/gongtiS/Data/NormalTarget/*csv')[47:55] #July 17-24
tplist = glob.glob(r'E:/gongtiS/Data/NormalTarget/*csv')[55:62] #July 25-31
hist = [pd.read_csv(i,header=None).as_matrix(columns=None) for i in hplist] #historical dataset
#vali = [pd.read_csv(i,header=None).as_matrix(columns=None) for i in vplist] #validation dataset
test = [pd.read_csv(i,header=None).as_matrix(columns=None) for i in tplist] #test dataset

print('preprocessing training data...')
trainX,trainY,trainY_nofilt = dp.predata(hist)

mlr = LinearRegression()
print('model training on trainX...')
start = time.time()
mlr.fit(trainX.reshape(trainX.shape[0],trainX.shape[1]*trainX.shape[2]),trainY)
end = time.time()

trainX, trainY, trainY_nofilt = [], [], []

print('preprocessing testing data...')
testX,testY,testY_nofilt = dp.predata(test)

print('#####model predicting on testX...#####')
predY = mlr.predict(testX.reshape(testX.shape[0],testX.shape[1]*testX.shape[2]))

MAE = np.mean(np.abs(np.asarray(predY).reshape(-1, link) * maxv - np.asarray(testY_nofilt).reshape(-1, link) * maxv))
MAPE = np.mean(np.abs(np.asarray(predY).reshape(-1, link) - np.asarray(testY_nofilt).reshape(-1, link)) / np.asarray(testY_nofilt).reshape(-1, link))
print('running time:', (end - start))
print('inputstep=%d' % inputstep)
print('predstep=%d' % predstep)
print('MAE:%.4f' % MAE)
print('MAPE:%.4f%%' % (MAPE * 100))

pd.DataFrame((np.asarray(predY).reshape(-1, link) * maxv), columns=None).to_csv(
    r'F:/[磕盐]服务器/NanJi/response/LR-prep%dp%d.csv' % (inputstep, predstep), header=None, columns=None)
with open(r'F:/[磕盐]服务器/NanJi/response/log-LR.txt', 'a') as f:
    f.write('\ninputstep=%d,predstep=%d,MAE=%.4f,MAPE=%.4f' % (inputstep, predstep, MAE, (MAPE * 100)))

testX, testY, testY_nofilt, predY = [], [], [], []


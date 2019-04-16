# -*- coding: utf-8 -*-
"""
@author: Nan Ji
"""

import numpy as np
import pandas as pd
import glob
import time
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from preprocess import DataPreprocess

dp = DataPreprocess(inputstep=6,predstep=0)
inputstep = dp.inputstep
predstep = dp.predstep
maxv = dp.maxv
link = 257

print('loading and partitioning dataset...')
hplist = glob.glob(r'E:/gongtiS/Data/NormalTarget/*csv')[30:47] #July 01-16
vplist = glob.glob(r'E:/gongtiS/Data/NormalTarget/*csv')[47:55] #July 17-24
tplist = glob.glob(r'E:/gongtiS/Data/NormalTarget/*csv')[55:62] #July 25-31
hist = [pd.read_csv(i,header=None).as_matrix(columns=None) for i in hplist] #historical dataset
vali = [pd.read_csv(i,header=None).as_matrix(columns=None) for i in vplist] #validation dataset
test = [pd.read_csv(i,header=None).as_matrix(columns=None) for i in tplist] #test dataset

print('data preprocessing...')
print('preprocessing validation data...')
valiX,valiY,valiY_nofilt = dp.predata(vali)

#n_folds = 5
rf = RandomForestRegressor()
#rf_vali = GridSearchCV(MultiOutputRegressor(rf),cv=None,
#                   param_grid=[{'estimator__n_estimators':range(20,101,10)}])
#print('model training on valilX...')
#rf_vali.fit(valiX[:,-1,:].astype('float32'),valiY)
#bestParams = rf_vali.best_params_
#print("Best parameters:",bestParams)
#valiX,valiY,valiY_nofilt = [],[],[]
bestParams = {'estimator__n_estimators':[100]}

"""
Model Training
"""
print('preprocessing train data...')
trainX,trainY,trainY_nofilt = dp.predata(hist)
params = {"estimator__n_estimators":[bestParams['estimator__n_estimators']]}
rf_test = GridSearchCV(MultiOutputRegressor(rf),param_grid=params)
print('model training on trainX...')
start = time.time()
rf_test.fit(trainX[:,-1,:],trainY)
end = time.time()

print('#####model predicting on trainX...#####')
tpredY = rf_test.predict(trainX[:,-1,:])
RMSE = math.sqrt(mean_squared_error(np.asarray(trainY_nofilt).reshape(-1,link)*maxv, np.asarray(tpredY).reshape(-1,link)*maxv))
MAPE = np.mean(np.abs(np.asarray(tpredY).reshape(-1,link)-np.asarray(trainY_nofilt).reshape(-1,link))/np.asarray(trainY_nofilt).reshape(-1,link))
print('running time:',(end-start).seconds)
print('inputstep=%d'%inputstep)
print('predstep=%d'%predstep)
print('RMSE:%.4f'%RMSE)
print('MAPE:%.4f%%'%(MAPE*100))
trainX,trainY,train_nofilt = [],[],[]


print('\n#####model predicting on testX...#####')
print('preprocessing test data...')
testX,testY,testY_nofilt = dp.predata(test)
predY = rf_test.predict(testX[:,-1,:])
RMSE = np.mean(np.sqrt(np.square(np.asarray(predY).reshape(-1,link)*maxv-np.asarray(testY_nofilt).reshape(-1,link)*maxv)))
MAPE = np.mean(np.abs(np.asarray(predY).reshape(-1,link)-np.asarray(testY_nofilt).reshape(-1,link))/np.asarray(testY_nofilt).reshape(-1,link))
print('inputstep=%d'%inputstep)
print('predstep=%d'%predstep)
print('RMSE:%.4f'%RMSE)
print('MAPE:%.4f%%'%(MAPE*100))

pd.DataFrame((np.asarray(predY).reshape(-1,link)*maxv),columns=None).to_csv(r'F:/NanJi/response/RF-prep%dp%d.csv'%(inputstep,predstep),header=None,columns=None)
with open(r'F:/NanJi/response/log-RF-prep%dp%d.txt'%(inputstep,predstep),'a') as f:
  f.write('\nRMSE=%.4f,MAPE=%.4f,best_parameters=%s'%(RMSE,MAPE*100,str(bestParams)))

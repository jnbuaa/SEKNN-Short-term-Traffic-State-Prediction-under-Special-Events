# -*- coding: utf-8 -*-
"""
@author: Nan Ji
"""

import numpy as np
import pandas as pd
import glob
import time
import warnings
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from preprocess import DataPreprocess

warnings.filterwarnings("ignore")

for i in [0]:
    dp = DataPreprocess(inputstep=6, predstep=i)
    inputstep = dp.inputstep
    predstep = dp.predstep
    maxv = dp.maxv
    link = 257

    print('loading and partitioning dataset...')
    hplist = glob.glob(r'E:/gongtiS/Data/NormalTarget/*csv')[30:47]  # July 01-16
    vplist = glob.glob(r'E:/gongtiS/Data/NormalTarget/*csv')[47:55]  # July 17-24
    tplist = glob.glob(r'E:/gongtiS/Data/NormalTarget/*csv')[55:62]  # July 25-31
    hist = [pd.read_csv(i, header=None).as_matrix(columns=None) for i in hplist]  # historical dataset
    vali = [pd.read_csv(i, header=None).as_matrix(columns=None) for i in vplist]  # validation dataset
    test = [pd.read_csv(i, header=None).as_matrix(columns=None) for i in tplist]  # test dataset

    """
    Parameters Calibrating
    """
    # print('preprocessing validation data...')
    # valiX,valiY,valiY_nofilt = dp.predata(vali)

    # n_folds = 5
    # grid_parameters = {'estimator__learning_rate':[1e-1,1e-2,1e-3,1e-4],
    #                    'estimator__n_estimators':range(20,101,10),'estimator__max_depth':range(3,14,2)}
    gbdt = GradientBoostingRegressor()
    # gbdt_vali = GridSearchCV(MultiOutputRegressor(gbdt),grid_parameters,cv=None)
    # gbdt_vali = gbdt_vali.fit(valiX[:,-1,:],valiY)
    # bestParams = gbdt_vali.best_params_
    # print("Best parameters:", bestParams)
    # valiX,valiY = [],[]

    bestParams = {'estimator__learning_rate': [1e-2],
                  'estimator__n_estimators': [100], 'estimator__max_depth': [5]}

    """
    Model Training
    """
    print('preprocessing train data...')
    trainX, trainY, trainY_nofilt = dp.predata(hist)
    params = {"estimator__learning_rate": [bestParams['estimator__learning_rate']],
              "estimator__n_estimators": [bestParams['estimator__n_estimators']],
              "estimator__max_depth": [bestParams['estimator__max_depth']]}
    gbdt_test = GridSearchCV(MultiOutputRegressor(gbdt), param_grid=bestParams)
    start = time.time()
    print('model training on trainX...')
    gbdt_test = gbdt_test.fit(trainX[:, -1, :], trainY)
    end = time.time()

    # print('#####model predicting on trainX...#####')
    # tpredY = gbdt_test.predict(trainX[:,-1,:])
    # RMSE = np.mean(np.sqrt(np.square(np.asarray(tpredY).reshape(-1,link)*maxv-np.asarray(trainY_nofilt).reshape(-1,link)*maxv)))
    # MAPE = np.mean(np.abs(np.asarray(tpredY).reshape(-1,link)-np.asarray(trainY_nofilt).reshape(-1,link))/np.asarray(trainY_nofilt).reshape(-1,link))
    # print('running time:',(end-start).seconds)
    # print('inputstep=%d'%inputstep)
    # print('predstep=%d'%predstep)
    # print('RMSE:%.4f'%RMSE)
    # print('MAPE:%.4f%%'%(MAPE*100))
    trainX, trainY, train_nofilt = [], [], []

    """
    Model Testing
    """
    print('preprocessing test data...')
    testX, testY, testY_nofilt = dp.predata(test)
    print('#####model predicting on testX...#####')
    predY = gbdt_test.predict(testX[:, -1, :])
    RMSE = np.mean(np.sqrt(
        np.square(np.asarray(predY).reshape(-1, link) * maxv - np.asarray(testY_nofilt).reshape(-1, link) * maxv)))
    MAPE = np.mean(
        np.abs(np.asarray(predY).reshape(-1, link) - np.asarray(testY_nofilt).reshape(-1, link)) / np.asarray(
            testY_nofilt).reshape(-1, link))
    print('running time:', (end - start))
    print('inputstep=%d' % inputstep)
    print('predstep=%d' % predstep)
    print('RMSE:%.4f' % RMSE)
    print('MAPE:%.4f%%' % (MAPE * 100))

    pd.DataFrame((np.asarray(predY).reshape(-1, link) * maxv), columns=None).to_csv(
        r'F:/[磕盐]服务器/NanJi/response/GBDT-prep%dp%d.csv' % (inputstep, predstep), header=None, columns=None)
    with open(r'F:/[磕盐]服务器/NanJi/response/log-GBDT.txt', 'a') as f:
        f.write('\ninputstep=%d,predstep=%d,RMSE=%.4f,MAPE=%.4f,best_parameters=%s' % (
        inputstep, predstep, RMSE, (MAPE * 100), str(bestParams)))

    testX, testY, testY_nofilt, predY = [], [], [], []

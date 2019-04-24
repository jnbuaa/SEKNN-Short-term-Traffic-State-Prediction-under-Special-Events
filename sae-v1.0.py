# -*- coding: utf-8 -*-
"""
@author: Nan Ji 
"""
import numpy as np
np.random.seed(1337)
import pandas as pd
import glob
import time
from preprocess import DataPreprocess
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

dp = DataPreprocess(inputstep=6,predstep=1)
inputstep = dp.inputstep
predstep = dp.predstep
maxv = dp.maxv
link = 257

print('loading and partitioning dataset...')
hplist = glob.glob(r'E:/gongtiS/Data/NormalTarget/*csv')[30:47] #July 01-16
tplist = glob.glob(r'E:/gongtiS/Data/NormalTarget/*csv')[55:62] #July 25-31
hist = [pd.read_csv(i,header=None).as_matrix(columns=None) for i in hplist] #historical dataset
test = [pd.read_csv(i,header=None).as_matrix(columns=None) for i in tplist] #test dataset

print('data preprocessing...')
trainX,trainY,trainY_nofilt = dp.predata(hist)
testX,testY,testY_nofilt = dp.predata(test)
trainX = trainX.reshape(trainX.shape[0],trainX.shape[1]*trainX.shape[2])
testX = testX.reshape(testX.shape[0],testX.shape[1]*testX.shape[2])

"""
Buliding a SAE
"""
def autoencoder(n_input,n_hidden,n_output):
    model = Sequential()
    model.add(Dense(n_hidden,input_dim=n_input,name='hidden'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(n_output))
    return model

def stacking_autoencoder(n_input,n_h1,n_h2,n_h3,n_output):
    ae1 = autoencoder(n_input,n_h1,n_output)
    ae2 = autoencoder(n_h1,n_h2,n_output)
    ae3 = autoencoder(n_h2,n_h3,n_output)
    
    sae = Sequential()
    sae.add(Dense(n_h1,input_dim=n_input,name='hidden_1'))
    sae.add(Activation('relu'))
    sae.add(Dense(n_h2,name='hidden_2'))
    sae.add(Activation('relu'))
    sae.add(Dense(n_h3,name='hidden_3'))
    sae.add(Activation('relu'))
    sae.add(Dropout(0.2))
    sae.add(Dense(n_output))
    
    models = [ae1,ae2,ae3,sae]
    return models

"""
Training
"""
models = stacking_autoencoder(inputstep*link,400,400,400,link)
temp = trainX
for i in range(len(models)-1):
    if i > 0:
        p = models[i-1]
        pre_model = Model(input=p.input,output=p.get_layer(name='hidden').output)
        temp = pre_model.predict(temp)
    
    m = models[i]
    m.compile(loss='mse',optimizer='rmsprop')
    m.fit(temp,trainY,batch_size=64,epochs=500)
    
    models[i] = m
    
sae = models[-1]
for i in range(len(models)-1):
    weights = models[i].get_layer(name='hidden').get_weights()
    sae.get_layer(name='hidden_%d'%(i+1)).set_weights(weights)


earlyStopping = EarlyStopping(monitor='val_loss',min_delta=0,patience=15,verbose=1)
modelCheckPoint = ModelCheckpoint(filepath=r'F:/NanJi/response/SAE%dp%d.h5'%(inputstep,predstep),monitor='val_loss',save_best_only=True,mode='min',verbose=0)
sae.compile(loss='mse',optimizer='rmsprop')
history = sae.fit(trainX,trainY,batch_size=64,epochs=2000,validation_split=0.05,callbacks=[earlyStopping,modelCheckPoint])

#history.loss_plot()
plt.plot(history.history['loss'],label='train',color='blue')
plt.plot(history.history['val_loss'],label='test',color='red')
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.legend()
plt.show()

print('\n#####model predicting on testX...#####')
predY = sae.predict(testX)
MAE = np.mean(np.abs(np.asarray(predY).reshape(-1, link) * maxv - np.asarray(testY_nofilt).reshape(-1, link) * maxv))
MAPE = np.mean(np.abs(np.asarray(predY).reshape(-1,link)-np.asarray(testY_nofilt).reshape(-1,link))/np.asarray(testY_nofilt).reshape(-1,link))
print('inputstep=%d'%inputstep)
print('predstep=%d'%predstep)
print('MAE:%.4f'%MAE)
print('MAPE:%.4f%%'%(MAPE*100))

pd.DataFrame((np.asarray(predY).reshape(-1,link)*maxv),columns=None).to_csv(r'F:/NanJi/response/SAE-prep%dp%d.csv'%(inputstep,predstep),header=None,columns=None)
with open(r'F:/NanJi/response/log-SAE-prep%dp%d.txt'%(inputstep,predstep),'a') as f:
    f.write('\ninputstep=%d,predstep=%d,MAE=%.4f,MAPE=%.4f'%(inputstep,predstep,MAE,MAPE*100))

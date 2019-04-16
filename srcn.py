# -*- coding: utf-8 -*-
"""
Created on Thu Apr 2 13:12:02 2019

@author: wuzhihai
"""

import numpy as np
np.random.seed(1337)
import pandas as pd
from preprocess import DataPreprocess
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Conv2D
from keras.layers import MaxPooling2D
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.optimizers import RMSprop
from keras.layers.wrappers import TimeDistributed
import matplotlib.pyplot as plt

"""
Note that:
    SRCN using the data from the grid-based network representation.
"""

dp = DataPreprocess(inputstep=15,predstep=0)
inputstep = dp.inputstep
predstep = dp.predstep
link = 257

model = Sequential()
model.add(TimeDistributed(Conv2D(16, (3,3), padding='same',kernel_initializer='glorot_uniform',name='conv_1'),
                          input_shape=(inputstep,1,162,224)))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2),padding='same')))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(TimeDistributed(Conv2D(32, (3,3), padding='same',kernel_initializer='glorot_uniform',name='conv_2')))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2),padding='same')))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(TimeDistributed(Conv2D(64, (3,3), padding='same',kernel_initializer='glorot_uniform',name='conv_3')))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(TimeDistributed(Conv2D(64, (3,3), padding='same',kernel_initializer='glorot_uniform',name='conv_4')))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(TimeDistributed(Conv2D(128,(3,3), padding='same',kernel_initializer='glorot_uniform',name='conv_5')))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2),padding='same')))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Reshape((inputstep,np.prod(model.output_shape[-3:]))))
model.add(TimeDistributed(Flatten()))
model.add(BatchNormalization())
model.add(LSTM(units=800,return_sequences=True))
model.add(Activation('tanh'))
model.add(LSTM(units=800,return_sequences=False))
model.add(Activation('tanh'))
model.add(Dropout(0.2))
model.add(Dense(link))

rmsprop = RMSprop()
model.compile(loss='mean_squared_error',optimizer=rmsprop)
print('Model has been compiled!')


"""
Training 
"""
print('loading training data...')
trainX = np.load('H:/ARN/trainX_out_of_order.npy').reshape(-1,inputstep,162,224,1)
trainY = np.load('H:/ARN/trainY_out_of_order.npy')

earlyStopping = EarlyStopping(monitor='val_loss',min_delta=0,patience=15,verbose=1)
modelCheckPoint = ModelCheckpoint(filepath='F:/NanJi/response/SRCN%dp%d.h5'%(inputstep,predstep),monitor='val_loss',save_best_only=True,mode='min',verbose=0)
print('Training...')
history = model.fit(trainX,trainY,batch_size=64,epochs=2000,validation_split=0.05,callbacks=[earlyStopping,modelCheckPoint])

#history.loss_plot()
plt.plot(history.history['loss'],label='train',color='blue')
plt.plot(history.history['val_loss'],label='test',color='red')
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.legend()
plt.show()

"""
Testing
"""
print('loading test data...')
trainX,trainY = [],[]
testX = np.load('H:/ARN/testX.npy').reshape(-1,inputstep,162,224,1)
testY = np.load('H:/ARN/testY.npy')
print('predicting...')
predY = model.predict(testX)
RMSE = np.mean(np.sqrt(np.square(np.asarray(predY).reshape(-1,link)-np.asarray(testY).reshape(-1,link))))
MAPE = np.mean(np.abs(np.asarray(predY).reshape(-1,link)-np.asarray(testY).reshape(-1,link))/np.asarray(testY).reshape(-1,link))
print('inputstep=%d'%inputstep)
print('predstep=%d'%predstep)
print('RMSE:%.4f'%RMSE)
print('MAPE:%.4f%%'%(MAPE*100))

pd.DataFrame((np.asarray(predY).reshape(-1,link)),columns=None).to_csv(r'F:/NanJi/response/SRCN-prep%dp%d.csv'%(inputstep,predstep),header=None,columns=None)
with open(r'F:/NanJi/response/log-SRCN.txt','a') as f:
    f.write('\ninputstep=%d,predstep=%d,RMSE=%.4f,MAPE=%.4f'%(inputstep,predstep,RMSE,MAPE*100))

# -*- coding: utf-8 -*-
"""
@author: Nan Ji
"""

import numpy as np 
np.random.seed(1337)
import pandas as pd
import scipy.signal as signal
import csv
import datetime

class DataPreprocess:
    
    def __init__(self,inputstep=6,predstep=0):
        self.inputstep = inputstep
        self.predstep = predstep
        self.rid, self.rclass = self.roadclass()
        self.maxv = self.maxspeed(self.rid,self.rclass)
     
    def roadclass(self):
        #classfile=pd.read_csv(self.filepath)
        print('loading road grade...')
        rclass,rid=[],[]
        with open(r'E:/gongtiS/gisData/roadclass.csv') as c:
            reader=csv.reader(c)
            for row in reader:
                try:
                    rid.append(int(row[0].split(';')[0]))
                    rclass.append(int(row[0].split(';')[-1][:5]))#first 5 chars of the 2rd str splited by ','
                except ValueError:
                    pass
        return rid,rclass

    def maxspeed(self,rid,rclass,count=False):
        print('calcuating max speed...')
        ms=[]
        w=0
        x=0
        y=0
        z=0
        for index_,rid_ in enumerate(rid):
            if rclass[index_]==41000:
                ms.append(120)
                w+=1
            elif rclass[index_]==42000:
                ms.append(80)
                x+=1
                print(rid_)
            elif rclass[index_]==43000:
                ms.append(80)
                y+=1
            else:
                ms.append(50)
                z+=1
        if count==False:
            return ms
        else:
            return ms,(w,x,y,z)
    
    
    def timelist(self):
        s='00:00:00'
        t=datetime.datetime.strptime(s,'%H:%M:%S')
        l=[]
        for i in range(720):
            l.append(datetime.datetime.strftime(t,'%H:%M:%S'))
            t=t+datetime.timedelta(minutes=2)
        return l
    
    def predata(self,data):
        dataX=[]
        dataY=[]
        dataY_nofilt=[]
        for i in range(len(data)):
            for j in range(720-self.inputstep-self.predstep):
                #filter
                data_filt=np.asarray([signal.medfilt(data[i][:,v],3) for v in range(len(data[i][0]))]).T
                #nomalization
                data_filt=np.asarray([data_filt[i,:]/np.asarray(self.maxv) for i in range(len(data_filt))])
                #adjustment
                data_filt[data_filt>1]=1
                dataX.append([data_filt[u] for u in range(j,j+self.inputstep)])
                dataY.append(data[i][j+self.inputstep+self.predstep]/np.asarray(self.maxv))
                dataY_nofilt.append(data[i][j+self.inputstep+self.predstep]/np.asarray(self.maxv))
        dataX = np.asarray(dataX)
        dataY = np.asarray(dataY)
        dataY_nofilt = np.asarray(dataY_nofilt)
        dataY_nofilt[dataY_nofilt == 0] = 1e4 
        return dataX,dataY,dataY_nofilt


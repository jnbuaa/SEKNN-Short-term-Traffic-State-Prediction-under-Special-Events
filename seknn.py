# -*- coding: utf-8 -*-
"""
@author: Nan Ji
"""
import pandas as pd
import glob
import numpy as np
import math
import datetime
import sys
from sklearn.metrics import mean_squared_error
from preprocess import DataPreprocess

dp = DataPreprocess(inputstep=6,predstep=0)
maxv = dp.maxv
inputstep = dp.inputstep
predstep = dp.predstep


print('corrlink...')
cl = pd.read_csv(r'E:/gongtiS/gisData/delcorrlink.csv',header=None)
cl = list(np.asarray(cl)[:,1])


print('loading and partitioning dataset...')
hplist = glob.glob(r'E:/gongtiS/Data/NormalTarget/*csv')[30:47] #July 01-16
#vplist = glob.glob(r'E:/gongtiS/Data/NormalTarget/*csv')[47:55] #July 17-24
tplist = glob.glob(r'E:/gongtiS/Data/NormalTarget/*csv')[55:62] #July 25-31
hist = [pd.read_csv(i,header=None).as_matrix(columns=None) for i in hplist] #historical dataset
#vali = [pd.read_csv(i,header=None).as_matrix(columns=None) for i in vplist] #validation dataset
test = [pd.read_csv(i,header=None).as_matrix(columns=None) for i in tplist] #test dataset


def trend(dataset):
    trend_trainX = []
    for i in range(len(dataset)):
        trend_trainX.append(list(map(lambda S1,S2:S2-S1,dataset[i][0],dataset[i][-1])))



def distance(x1,x2,dtype='ed'):
    """
    cos*ed = cos(trend_x1,trend_x2)*Edist(x1,x2)
    """
    x1,x2 = np.asarray(x1),np.asarray(x2)
    if dtype == 'cos':
#        trend1trend2=np.dot(trend1,trend2)
#        trend1mo=math.sqrt(np.dot(trend1,trend1))
#        trend2mo=math.sqrt(np.dot(trend2,trend2))
#        cos=trend1trend2/(trend1mo*trend2mo)
        cos = np.dot(x1,x2)/(math.sqrt(np.dot(x1,x1))*math.sqrt(np.dot(x2,x2)))
        #print(cos)
        return cos
    elif dtype == 'ed': #Euclidean distance
        #ed=math.sqrt(np.dot(x1-x2,x1-x2))
        ed = np.linalg.norm(x1-x2,ord=2)
        return ed
    elif dtype == 'kl': #KL-divergence
        x1 = np.asarray(x1)
        x2 = np.asarray(x2)
        return abs(np.sum(x2*np.log(x1/x2)))
    elif dtype == 'hellinger': #hellinger distance/#Bhattacharyya Distance
        x1 = np.asarray(x1)
        x2 = np.asarray(x2)
        return np.sqrt(1-np.sum(np.sqrt(x1*x2)))
    elif dtype == 'bc': #Bhattacharyya Distance
        x1 = np.asarray(x1)
        x2 = np.asarray(x2)
        return np.sqrt(1-np.sum(np.sqrt(x1*x2)))
#    elif dtype=='md': #mahalanobis distance
#        return np.sqrt(np.dot(np.dot((x1-x2).reshape(1,link),np.linalg.inv(cov)),(x1-x2).reshape(link,1)))

def increment(trainX,trainY):
    increment = []
    for i in range(len(trainY)):
        increment.append(trainY[i]-trainX[i][-1])
    return increment

def quicksort(array):  
    if len(array) < 2:  
        return array  
    else:  
        pivot = array[0]  
        less = [i for i in array[1:] if i <= pivot]  
        rather = [i for i in array[1:] if i > pivot]  
    return quicksort(less) + [pivot] + quicksort(rather)  
  
def sort_index(l):  
    index = {}  
    for i in range(len(l)):  
        index[l[i]] = i
    return index  
  
def sort(list_sort):  
    index_of_list = sort_index(list_sort)  
    sort = quicksort(list_sort)  
    # print(sorted)  
    rank = []
    for i in sort:  
        rank.append(index_of_list[i])  
    #print('Ð¡´óÅÅÐò(°´Ë÷Òý´ÓÐ¡µ½´ó):',list(rank))
    return rank


predsteps = [0,1,2,3,4]
inputstep = 6#[2,3,4,5,6,7,8,9,10]
link = 257

parameters = [0.9,0.9,0.8,0.8,0.7]
thersholds = [97,57,57,54,54]
alpha = parameters[predsteps.index(predstep)]
k = thersholds[predsteps.index(predstep)]  

print('data preprocessing...')
trainX,trainY,trainY_nofilt = dp.predata(hist)
#valiX,valiY,valiY_nofilt = predata(vali,inputsteps,predstep)
testX,testY,testY_nofilt = dp.predata(test)
trainX_corr = np.asarray(trainX)[:,:,cl]
#valiX_corr = np.asarray(valiX)[:,:,cl]
testX_corr = np.asarray(testX)[:,:,cl]
#pd.DataFrame(np.asarray(valiY_nofilt)*maxv).to_csv(r'E:/gongtiS/result/reexperiment/valiY%dp%d.csv'%(inputsteps,predstep),columns=None,header=None)
pd.DataFrame(np.asarray(testY_nofilt)*maxv).to_csv(r'E:/gongtiS/result/reexperiment/testY%dp%d.csv'%(inputstep,predstep),columns=None,header=None)


print('calcuating trend vectors...')
trend_trainX = []
#trend_valiX = []
trend_testX = []
for i in range(len(trainX_corr)):
    trend_trainX.append(list(map(lambda S1,S2:S2-S1,trainX_corr[i][0],trainX_corr[i][-1])))
#for i in range(len(valiX_corr)):
#    trend_valiX.append(list(map(lambda S1,S2:S2-S1,valiX_corr[i][0],valiX_corr[i][-1])))
for i in range(len(testX_corr)):
    trend_testX.append(list(map(lambda S1,S2:S2-S1,testX_corr[i][0],testX_corr[i][-1])))

print('calcuating the increment...')    
incr = increment(trainX,trainY)

tl = dp.timelist()
tpred = tl[predstep:]

"""
vaildate
"""
print('experiment on validation dataset...')
#parameter=[0.9]#[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
#thershold=[97]#[55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160]#[1,5,10,15,20,25,30,35,40,45,50,55,60,
rmse,mape = [],[]
pred = []
start = datetime.datetime.now()
for ind_a,a in enumerate(testX_corr):
    ED = []
    cosTrD = []
    d = [] #dist results
    for ind_x,x in enumerate(trainX_corr):
        ED.append(distance(x[-1],a[-1],dtype='ed'))
        cosTrD.append(1-distance(trend_trainX[ind_x],trend_testX[ind_a],dtype='cos'))
    maxED = max(ED)
    minED = min(ED)
    ED01 = [2*(d-minED)/(maxED-minED) for d in ED]
    d = list(map(lambda i,j:alpha*i+(1-alpha)*j,ED01,cosTrD))
    order_alld = sort(d) #return sorted index
    topk_d = order_alld[:k]
    #incr for SD-KNN; trainY for original KNN
    topk_nn = np.asarray(incr)[topk_d].reshape(-1,link)
    #mean nearest neighbors
    #pred.append(np.mean(topk_nn,0))
    #Gaussian weighted nearest neighbors
    w=[math.e**(-(d[i])**2/(2*1.33**2)) for i in topk_d]
    pred.append(np.dot(w,topk_nn+testX[ind_a][-1])/np.sum(w))

    view='>'*(ind_a//200)
    sys.stdout.write('\r'+view+'[%.2f%%]'%(100*ind_a/len(testX_corr)))
    sys.stdout.flush()
    
end=datetime.datetime.now()
RMSE = np.mean(np.sqrt(np.square(np.asarray(pred).reshape(-1,link)*maxv-np.asarray(testY_nofilt).reshape(-1,link)*maxv)))
MAPE = np.mean(np.abs(np.asarray(pred).reshape(-1,link)-np.asarray(testY_nofilt).reshape(-1,link))/np.asarray(testY_nofilt).reshape(-1,link))
rmse.append(RMSE)
mape.append(MAPE)
print('running time:',(end-start).seconds)
print('alpha=%.1f'%alpha)
print('k=%d'%k)
print('inputstep=%d'%inputstep)
print('predstep=%d'%predstep)
print('RMSE:%.4f'%RMSE)
print('MAPE:%.4f%%'%(MAPE*100))

pd.DataFrame((np.asarray(pred).reshape(-1,link)*maxv),columns=None).to_csv(r'E:/gongtiS/result/reexperiment/prep%dp%d_k%d_a%d.csv'%(inputstep,predstep,k,alpha*10),header=None,columns=None)

f = open(r'E:/gongtiS/result/reexperiment/logcorr_%dp%d.txt'%(inputstep,predstep),'a')
f.write('\nalpha=%.1f,k=%d,RMSE=%.4f,MAPE=%.4f'%(alpha,k,RMSE,MAPE*100))
f.close

trainX,trainY,trainY_nofilt,trainX_corr=[],[],[],[]
#valiX,valiY,valiY_nofilt,valiX_corr=[],[],[],[]
testX,testY,testY_nofilt,testX_corr=[],[],[],[]

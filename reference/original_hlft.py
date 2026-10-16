import sys
import pandas as pd
from numba import jit
import numpy as np
from numba.typed import List
from math import *
import math
import time
#隐特征矩阵的维度
rank=20
max_iteration=1000 #最大迭代次数
#训练，验证，测试集的路径
trainName="C:\\Users\\XXX\\Desktop\\Lp范数源码\\LPSGD\\data\\Guangzhou_data_set\\Guangzhou_train70.txt"
validName="C:\\Users\\XXX\\Desktop\\Lp范数源码\\LPSGD\\data\\Guangzhou_data_set\\Guangzhou_valid10.txt"
testName="C:\\Users\\XXX\\Desktop\\Lp范数源码\\LPSGD\\data\\Guangzhou_data_set\\Guangzhou_test20.txt"
#张量三个维度的最大值和最小值
minAID=sys.maxsize
minBID=sys.maxsize
minCID=sys.maxsize
maxAID=0
maxBID=0
maxCID=0
#读取数据集
def dataSetGenerator(fileName):
    global minAID,minBID,maxAID,maxBID,maxCID,minCID
    dataSet=list()
    with open(fileName,'r') as f:
        for line in f.readlines():
            aid=int(float(line[:-1].split('::')[0])) #第一个维度的位置
            bid=int(float(line[:-1].split('::')[1]))#第二个维度的位置
            cid=int(float(line[:-1].split('::')[2]))#第三个维度的位置
            value=float(line[:-1].split('::')[-1])#张量的值
            if math.isnan(value) or value == 0:
                continue  # 跳过这一条
            if fileName==trainName:
                value=np.log10(value+1)
            dataSet.append(np.array([aid,bid,cid,value]))
            #求三个维度的最大最小值
            if aid>maxAID:
                maxAID=aid
            if bid>maxBID:
                maxBID=bid
            if cid>maxCID:
                maxCID=cid
            if aid<minAID:
                minAID=aid
            if bid<minBID:
                minBID=bid
            if cid<minCID:
                minCID=cid
    f.close()
    return dataSet
trainlist=dataSetGenerator(trainName)
#
# print('maxAID='+str(maxAID)+';maxBID='+str(maxBID)+';maxCID='+str(maxCID))
# print('minAID='+str(minAID)+';minBID='+str(minBID)+';minCID='+str(minCID))
@jit(nopython=True)
#将数据集转化为数组
def listtoarr(listtype):
    length=len(listtype)
    array=np.zeros((length,4))
    for num in range(length):
        for col in range(4):
            array[num][col]=listtype[num][col]
    return array
trainarray=listtoarr(trainlist)
validlist=dataSetGenerator(validName)
validarray=listtoarr(validlist)
testlist=dataSetGenerator(testName)
testarray=listtoarr(testlist)

@jit(nopython=True)
#初始化三个因子矩阵的梯度
def initgrad():
    grad_S = np.zeros((maxAID + 1, rank), dtype=np.float64)
    grad_D = np.zeros((maxBID + 1, rank), dtype=np.float64)
    grad_T = np.zeros((maxCID + 1, rank), dtype=np.float64)
    return grad_S, grad_D, grad_T

@jit(nopython=True)
#寻找张量值的最大值
def findmaxvalue():
    phi = 0
    for trainTensor in trainarray:
        if (trainTensor[3] > phi):
            phi = trainTensor[3]
    return phi

import math
import numpy as np

@jit(nopython=True)
def norm(x):
    return np.dot(x, x)

@jit(nopython=True)
def artanh(x):
    return 0.5 * np.log((1 + x) / (1 - x))

@jit(nopython=True)
#将因子矩阵向量的值限制在poincare球里面
def update(emb, grad, lr):
    eps = 1e-5
    c_ = (1 - norm(emb)) ** 2 / 4
    upd = lr * c_ * grad
    emb = emb - upd
    if (norm(emb) >= 1):
        emb = emb / sqrt(norm(emb)) - eps
    return emb


@jit(nopython=True)
#计算三个向量两两之间的距离之和
def computeDijk(a, b, c):
    Dijk = 0
    Dijk = np.arccosh(dist1(a, b)) + np.arccosh(dist1(a, c)) + np.arccosh(dist1(b, c))
    return Dijk

@jit(nopython=True)
#计算预测值
def getprediction(S, D, T, aid, bid, cid,phi):
    p_value = 0
    Dijk = computeDijk(S[aid], D[bid], T[cid])
    p_value = phi * (1 - tanh(Dijk))
    return p_value

@jit(nopython=True)
#定义两个向量之间的双曲空间距离
def dist1(vec1, vec2):
    diff_vec = vec1 - vec2
    return 1 + 2 * norm(diff_vec) / ((1 - norm(vec1)) * (1 - norm(vec2)))

@jit(nopython=True)
#计算双曲空间距离相对于每个向量的梯度
def compute_distance_gradients(theta, x, gamma):
    alpha = (1.0 - np.dot(theta, theta))
    norm_x = norm(x)
    beta = (1 - norm_x)
    c_ = 4.0 / (alpha * beta * sqrt(gamma ** 2 - 1))
    return c_ * ((norm_x - 2 * np.dot(theta, x) + 1) / alpha * theta - x)

@jit(nopython=True)
def grad_pred_D(a, b, c):
    Dijk = computeDijk(a, b, c)
    return -phi * (1 - tanh(Dijk) ** 2)
#初始化因子矩阵的值
def initpara(lh):
    S=np.random.uniform(low=-lh,high=lh,size=(maxAID+1,rank))
    D=np.random.uniform(low=-lh,high=lh,size=(maxBID+1,rank))
    T=np.random.uniform(low=-lh,high=lh,size=(maxCID+1,rank))
    return S,D,T
@jit(nopython=True)
#计算自适应步长
def adaptive_step_size(U,Id,lr0):
    lr=0
    norm_u=norm(U[Id])
    lr=lr0*np.exp(-sqrt(norm_u))
    return lr

@jit(nopython=True)
def Rund_update(S, D, T,phi, lr0, trainarray,validarray, testarray):
    grad_S, grad_D, grad_T = initgrad()
    #遍历训练集的每一个元素
    for trainTensor in trainarray:
        aid = int(trainTensor[0])
        bid = int(trainTensor[1])
        cid = int(trainTensor[2])
        p_value = getprediction(S, D, T, aid, bid, cid,phi)
        #计算误差
        error = trainTensor[3] - p_value
        #计算双曲空间距离相对于每个因子矩阵的梯度
        grad_g_D = -2 * error * grad_pred_D(S[aid], D[bid], T[cid])
       
        grad_S[aid] = compute_distance_gradients(S[aid], D[bid], dist1(S[aid], D[bid])) + \
                      compute_distance_gradients(S[aid], T[cid], dist1(S[aid], T[cid]))
        grad_D[bid] = compute_distance_gradients(D[bid], S[aid], dist1(S[aid], D[bid])) + \
                      compute_distance_gradients(D[bid], T[cid], dist1(D[bid], T[cid]))
        grad_T[cid] = compute_distance_gradients(T[cid], S[aid], dist1(S[aid], T[cid])) + \
                      compute_distance_gradients(T[cid], D[bid], dist1(D[bid], T[cid]))
        #更新因子矩阵S的学习率
        s_aid_lr = adaptive_step_size(S, aid,lr0)
        # 更新因子矩阵
        S[aid] = update(S[aid], grad_g_D * grad_S[aid], s_aid_lr)
        #更新因子矩阵D的学习率
        d_bid_lr = adaptive_step_size(D, bid,lr0)
        #更新因子矩阵D
        D[bid] = update(D[bid], grad_g_D * grad_D[bid], d_bid_lr)
        #更新因子矩阵T的学习率
        t_cid_lr = adaptive_step_size(T, cid,lr0)
        #更新因子矩阵T
        T[cid] = update(T[cid], grad_g_D * grad_T[cid], t_cid_lr)

    square = 0
    absCount = 0
    mapeCount=0
    square2 = 0
    absCount2 = 0
    mapeCount2=0
    #计算验证集RMSE，MAE和MAPE
    for validTensor in validarray:
        aid = int(validTensor[0])
        bid = int(validTensor[1])
        cid = int(validTensor[2])
        p_value = getprediction(S, D, T, aid, bid, cid,phi)
        p_real=10**(p_value)-1
        square += (validTensor[3] - p_real) ** 2
        absCount += np.absolute(validTensor[3] - p_real)
        mapeCount += np.absolute((validTensor[3] - p_real)/validTensor[3])
    if ((len(validarray) > 0)):
        RMSE_valid = np.sqrt(np.divide(square, len(validarray)))
        MAE_valid = np.divide(absCount, len(validarray))
        MAPE_valid = np.divide(mapeCount, len(validarray))
    #计算测试集的RMSE,MAE和MAPE
    for testTensor in testarray:
        aid = int(testTensor[0])
        bid = int(testTensor[1])
        cid = int(testTensor[2])
        p_value = getprediction(S, D, T, aid, bid, cid,phi)
        p_real=10**(p_value)-1
        square2 += (testTensor[3] - p_real) ** 2
        absCount2 += np.absolute(testTensor[3] - p_real)
        mapeCount2 += np.absolute((testTensor[3] - p_real)/testTensor[3])
    if ((len(testarray) > 0)):
        RMSE_test = np.sqrt(np.divide(square2, len(testarray)))
        MAE_test = np.divide(absCount2, len(testarray))
        MAPE_test = np.divide(mapeCount2, len(testarray))
    return S,D,T,RMSE_valid, MAE_valid, MAPE_valid, RMSE_test, MAE_test, MAPE_test


def process(S, D, T,phi,lr0,trainarray,validarray, testarray):
    errorgap = 1e-5
    tr = 0
    flgRMSE = True
    flgMAE = True
    flgMAPE=True
    minRMSE = 100
    minMAE = 100
    minMAPE=100
    #定义三个指标的轮数
    RMSE_round = np.zeros(max_iteration + 1, dtype=np.float64)
    MAE_round = np.zeros(max_iteration + 1, dtype=np.float64)
    MAPE_round = np.zeros(max_iteration + 1, dtype=np.float64)
    RMSE_round2 = np.zeros(max_iteration + 1, dtype=np.float64)
    MAE_round2 = np.zeros(max_iteration + 1, dtype=np.float64)
    MAPE_round2 = np.zeros(max_iteration + 1, dtype=np.float64)
    RMSE_round2[0] = 100
    MAE_round2[0] = 100
    MAPE_round2[0] = 100
    minRMSERound = 0
    minMAERound = 0
    minMAPERound = 0
    #定义三个指标结束的时间
    endRMSETime = 0
    endMAETime = 0
    endMAPETime = 0
    RMSETime = 0
    MAETime = 0
    MAPETime = 0
    for n in range(max_iteration):
        S,D,T,RMSE_valid, MAE_valid, MAPE_valid,RMSE_test, MAE_test, MAPE_test = \
            Rund_update(S, D, T,phi, lr0, trainarray,validarray,testarray)
        #保存每一轮验证集的三个指标
        RMSE_round[n] = RMSE_valid
        MAE_round[n] = MAE_valid
        MAPE_round[n]=MAPE_valid
        #保存每一轮测试集的三个指标
        RMSE_round2[n] = RMSE_test
        MAE_round2[n] = MAE_test
        MAPE_round2[n]= MAPE_test
        #print(n, RMSE_round2[n], MAE_round2[n])
      #计算RMSE的最小值，最小轮数和结束时间
        if (RMSE_round2[n - 1] - RMSE_round2[n] > errorgap):
            if (RMSE_round2[n] < minRMSE):
                minRMSE = RMSE_round2[n]
                minRMSERound = n
                endRMSETime = time.time()
            tr = 0
            flgRMSE = False
        # 计算MAE的最小值，最小轮数和结束时间
        if (MAE_round2[n - 1] - MAE_round2[n] > errorgap):
            if (MAE_round2[n] < minMAE):
                minMAE = MAE_round2[n]
                minMAERound = n
                endMAETime = time.time()
            tr = 0
            flgMAE = False
        # 计算MAPE的最小值，最小轮数和结束时间
        if (MAPE_round2[n - 1] - MAPE_round2[n] > errorgap):
            if (MAPE_round2[n] < minMAPE):
                minMAPE = MAPE_round2[n]
                minMAPERound = n
                endMAPETime = time.time()
            tr = 0
            flgMAPE = False

        #判断是否早停
        if flgRMSE and flgMAE and flgMAPE:
            tr = tr + 1
            if (tr == 5):
                RMSETime = endRMSETime - startTime
                # print(RMSETime,MAETime)
                MAETime = endMAETime - startTime
                MAPETime = endMAPETime - startTime
                break
        flgRMSE = True
        flgMAE = True
        flgMAPE= True
    return S,D,T,minRMSE,minMAE,minMAPE,minRMSERound, minMAERound, minMAPERound,RMSETime, MAETime, MAPETime


lr0list=[1e-3]#[1e-3,1e-4,1e-5]# 波兰1e-5
lh1=0.001#0.001
k=1
for i in range(1):
 for lr0 in lr0list:
    print(f"lr0={lr0}")
    trainarray_copy = np.copy(trainarray)
    validarray_copy = np.copy(validarray)
    testarray_copy = np.copy(testarray)
    #初始化三个因子矩阵的值
    Si, Di, Ti = initpara(lh1)
    #找张量里面的最大值
    phi = findmaxvalue()
    startTime = time.time()
    S,D,T,minRMSE, minMAE, minMAPE,minRMSERound, minMAERound, minMAPERound, RMSETime, MAETime,MAPETime = \
        process(Si, Di, Ti,phi, lr0,trainarray_copy,validarray_copy,testarray_copy)
    endTime = time.time()
    print("最小RMSE：" + str(minRMSE) + "\t" + "RMSE轮数:" + str(minRMSERound) + "\t" + "RMSE时间:" + str(RMSETime) + "\t")
    print("最小MAE： " + str(minMAE) + "\t" + " MAE轮数:" + str(minMAERound) + "\t" + "MAE时间:" + str(MAETime) + "\t")
    print("最小MAPE：" + str(minMAPE) + "\t" + "MAPE轮数:" + str(minMAPERound) + "\t" + "MAPE时间:" + str(MAPETime) + "\t")
   

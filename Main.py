import numpy as np
import os
import cv2 as cv
from numpy import matlib
import random as rn
from CSA import CSA
from CWO import CWO
from Global_Vars import Global_Vars
from Model_ANN import Model_ANN
from Model_H_DA_AD_DN import Model_H_DA_AD_DN
from Model_LSTM_GRU import Model_LSTM_GRU
from Model_YOLOv3_Mask_RCNN import Model_YOLOv3_Mask_RCNN
from NGO import NGO
from New_Plot import Plot_Results_new
from Obj_Fun import Obj_fun_pred
from PROPOSED import PROPOSED
from RMn_SSD_Datas import RMn_SSD_Feat
from Plot_Results import Plot_Results, Plot_table, Plot_Fitness
from SCOA import SCOA
from SSD_Mobilenet import RMn_SSD_Images

# Read Dataset1
an = 0
if an == 1:
    dir = './Dataset/Dataset1/train/'
    dir_list = os.listdir(dir)
    images = []
    for i in range(len(dir_list)):
        print(i)
        file = dir + dir_list[i]
        if '.txt' in dir_list[i]:
            continue
        else:
            read = cv.imread(file)
            read = cv.resize(read, [256, 256])
            images.append(read)
    np.save("Images_1.npy", images)

# Read Dataset2
an = 0
if an == 1:
    dir = './Dataset/Dataset2/train/images/'
    dir_list = os.listdir(dir)
    images = []
    for i in range(5000):
        print(i)
        file = dir + dir_list[i]
        read = cv.imread(file)
        read = cv.resize(read, [256, 256])
        images.append(read)
    np.save("Images_2.npy", images)

# Feature Extraction using Recurrent Mobilenet- Single Shot Detector
an = 0
if an == 1:
    for a in range(2):
        Images = np.load(f'Images_{a + 1}.npy', allow_pickle=True)
        labels = []
        detected_images = []
        for b in range(len(Images)):
            print(a, b, len(Images))
            image = cv.resize(Images[b], [1024, 1024])
            detected_image, vehicle_count = RMn_SSD_Images(image)
            detected_image = cv.resize(detected_image, [256, 256])
            if vehicle_count <= 2:
                label = "Low"
            elif vehicle_count > 2 and vehicle_count <= 5:
                label = "Medium"
            else:
                label = "High"
            detected_images.append(detected_image)
            labels.append(label)
        labels = np.asarray(labels)
        uni = np.unique(labels)
        tar = np.zeros((len(labels), len(uni))).astype('int')
        for c in range(len(uni)):
            ind = np.where(labels == uni[c])
            tar[ind[0], c] = 1
        np.save(f'Feature1_{a + 1}.npy', np.asarray(detected_images))
        np.save(f'Labels_{a + 1}.npy', labels)
        np.save(f'Targets_{a + 1}.npy', tar)

# Feature Extraction
an = 0
if an == 1:
    for a in range(2):
        Images = np.load(f'Images_{a + 1}.npy', allow_pickle=True)
        Feat = []
        for b in range(len(Images)):
            print(a, b)
            image = cv.resize(Images[b], [1024, 1024])
            vehicle_count, density, speed, lane_estimate = RMn_SSD_Feat(image)
            Feat.append([vehicle_count, density, speed, lane_estimate])
        np.save(f'Feature2_{a + 1}.npy', np.asarray(Feat, np.dtype(object)))

# Optimization for prediction
an = 0
if an == 1:
    fitness = []
    Bestsol = []
    for a in range(2):
        Feat1 = np.load(f'Feature1_{a + 1}.npy', allow_pickle=True)
        Feat2 = np.load(f'Feature2_{a + 1}.npy', allow_pickle=True)
        Target = np.load(f'Targets_{a + 1}.npy', allow_pickle=True)
        Npop = 10
        Chlen = 4
        xmin = matlib.repmat([5, 0.01, 100], Npop, 1)
        xmax = matlib.repmat([255, 0.99, 500], Npop, 1)
        initsol = np.zeros((Npop, Chlen))
        for p1 in range(initsol.shape[0]):
            for p2 in range(initsol.shape[1]):
                initsol[p1, p2] = rn.uniform(xmin[p1, p2], xmax[p1, p2])
        Global_Vars.Feat1 = Feat1
        Global_Vars.Feat2 = Feat2
        Global_Vars.Target = Target
        fname = Obj_fun_pred
        Max_iter = 50

        print("CSA...")
        [bestfit1, fitness1, bestsol1, time1] = CSA(initsol, fname, xmin, xmax, Max_iter)

        print("WOA...")
        [bestfit2, fitness2, bestsol2, time2] = NGO(initsol, fname, xmin, xmax, Max_iter)

        print("BBFGO...")
        [bestfit3, fitness3, bestsol3, time3] = CWO(initsol, fname, xmin, xmax, Max_iter)

        print("ABOA...")
        [bestfit4, fitness4, bestsol4, time4] = SCOA(initsol, fname, xmin, xmax, Max_iter)

        print("PROPOSED...")
        [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)

        Best = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
        Fit = ([fitness1.ravel(), fitness2.ravel(), fitness3.ravel(), fitness4.ravel(), fitness5.ravel()])
        Bestsol.append(Best)
        fitness.append(Fit)
    np.save('Bestsol_pred.npy', np.asarray(Bestsol))
    np.save('Fitness.npy', np.asarray(fitness))

# Prediction
an = 0
if an == 1:
    Batch_size = [4, 8, 16, 32, 48]
    Eval_all = []
    for a in range(2):
        bests = np.load('Bestsol_pred.npy', allow_pickle=True)[a]
        Feat1 = np.load(f'Feature1_{a + 1}.npy', allow_pickle=True)
        Feat2 = np.load(f'Feature2_{a + 1}.npy', allow_pickle=True)
        Target = np.load(f'Targets_{a + 1}.npy', allow_pickle=True)
        EVAL = []
        Data = np.concatenate(Feat1.reshape(Feat1.shape[0], Feat1.shape[1:].ravel(), Feat2), axis=1)
        for i in range(len(Batch_size)):
            Eval = np.zeros((10, 14))
            learnper = round(Data.shape[0] * Batch_size[i])
            for j in range(bests.shape[1]):
                sol = bests[j]
                Train_Data = Data[:learnper, :]
                Train_Target = Target[:learnper, :]
                Test_Data = Data[learnper:, :]
                Test_Target = Target[learnper:, :]
                Eval[j, :] = Model_H_DA_AD_DN(Feat1, Feat2, Target, sol)
            Train_Data = Data[:learnper, :]
            Train_Target = Target[:learnper, :]
            Test_Data = Data[learnper:, :]
            Test_Target = Target[learnper:, :]
            Eval[5, :] = Model_ANN(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[6, :] = Model_LSTM_GRU(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[7, :] = Model_YOLOv3_Mask_RCNN(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[8, :] = Model_H_DA_AD_DN(Feat1, Feat2, Target)
            EVAL.append(Eval)
        Eval_all.append(EVAL)
    np.save('Eval_Pred.npy', np.asarray(Eval_all))

Plot_Results()
Plot_table()
Plot_Fitness()
Plot_Results_new()

'''
ocsvm

author: lizhijian
date: 2024-05-31
'''

import sys
import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn import svm
import joblib
import cv2
import numpy as np
from tqdm import tqdm


class OCSVM(object):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = models.resnet50(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-1])  # remove the classification layer
        self.model.to(self.device)
        self.model.eval()
        self.ss = StandardScaler()
        self.ocsvmclf = svm.OneClassSVM(gamma=0.001,
                               kernel='rbf',
                               nu=0.08)
        self.ifclf = IsolationForest(contamination=0.08,
                            max_features=1.0,
                            max_samples=1.0,
                            n_estimators=40)
        self.pca = None

    def extractResnet(self, X):
        # X numpy array
        X = torch.from_numpy(X).float().to(self.device)
        with torch.no_grad():
            fe_array = self.model(X).cpu().numpy()
        return fe_array

    def prepareData(self, path):
        datalist = glob.glob(path+'/*.png')
        felist = []
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        for p in tqdm(datalist):
            img = cv2.imread(p)
            img = transform(img).unsqueeze(0).numpy()
            fe = self.extractResnet(img)
            felist.append(fe.reshape(1,-1))
        
        X_t = felist[0]
        for i in range(len(felist)):
            if i == 0:
                continue
            X_t = np.r_[X_t, felist[i]]
        
        return X_t

    def initPCA(self, X_train):
        self.pca = PCA(n_components=X_train.shape[0], whiten=True)

    def doSSFit(self, Xs):
        self.ss.fit(Xs)

    def doPCAFit(self, Xs):
        self.pca = self.pca.fit(Xs)
        return Xs
    
    def doSSTransform(self, Xs):
        Xs = self.ss.transform(Xs)
        return Xs
    
    def doPCATransform(self, Xs):
        Xs = self.pca.transform(Xs)
        return Xs

    def train(self, Xs):
        self.ocsvmclf.fit(Xs)
        self.ifclf.fit(Xs)

    def predict(self, Xs):
        pred = self.ocsvmclf.predict(Xs)
        return pred


def trainSVM():
    f = OCSVM()
    X_train = f.prepareData('data/train')
    # do StandardScaler
    f.doSSFit(X_train)
    X_train = f.doSSTransform(X_train)
    # do pca
    f.initPCA(X_train)
    f.doPCAFit(X_train)
    X_train = f.doPCATransform(X_train)
    # train svm
    f.train(X_train)
    
    # save our models
    joblib.dump(f.ocsvmclf, 'ocsvmclf.model')
    joblib.dump(f.pca, 'pca.model')
    joblib.dump(f.ss,'ss.model')

def loadSVMAndPredict():
    f = OCSVM()
    # load models
    f.ocsvmclf = joblib.load('ocsvmclf.model')
    f.pca = joblib.load('pca.model')
    f.ss = joblib.load('ss.model')

    X_test = f.prepareData('data/test')
    # do test data ss
    X_test = f.doSSTransform(X_test)
    # do test data pca
    X_test = f.doPCATransform(X_test)

    # predict
    preds = f.predict(X_test)
    print(f'{preds}')


if __name__ == '__main__':
    trainSVM()
    loadSVMAndPredict()
    pass
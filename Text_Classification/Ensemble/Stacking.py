import pickle
import pandas as pd
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB

def Load_Data(times):
    file = open('Sparse_Matrix.pkl','rb')
    train_XA = pickle.load(file)
    test_XA = pickle.load(file)
    train_XB = pickle.load(file)
    test_XB = pickle.load(file)
    file.close()
    train_X = sparse.hstack([train_XA, train_XB*times])
    test_X = sparse.hstack([test_XA, test_XB*times])
    train_y = pd.read_csv('train.csv')['pred']
    print("Load Data Complete!")
    return train_X, test_X,train_y

def Split_Data(train_X):
    kf = KFold(n_splits = 5, shuffle=False)
    train_id = []
    test_id = []
    for train_index , test_index in kf.split(train_X):
        train_id.append(train_index)
        test_id.append(test_index)
    file = open("Ensemble_Data/Split_index.pkl",'wb')
    pickle.dump(train_id,file)
    pickle.dump(test_id,file)
    file.close()
    print("Split K-fold complete")
    return train_id, test_id

def Load_id():
    file = open("Ensemble_Data/Split_index.pkl",'rb')
    train_id = pickle.load(file)
    test_id = pickle.load(file)
    file.close()
    return train_id, test_id

def Stacking_LR(train_X, test_X, train_y, train_id, test_id):
    for i in range(5):
        print("part %d begin"%i)
        id_A = train_id[i]
        id_B = test_id[i]
        trainX = train_X[id_A]
        testX = train_X[id_B]
        trainy = train_y[id_A]

        model = LogisticRegression(C=1.2, penalty='l1')
        model.fit(trainX, trainy)
        pred = model.predict_proba(testX)
        test_y = model.predict_proba(test_X) 
        with open("Ensemble_Data/Stack_LR_train%d.pkl"%i,'wb') as f1 :
            pickle.dump(pred, f1)
        with open("Ensemble_Data/Stack_LR_test%d.pkl"%i,'wb') as f2:
            pickle.dump(test_y, f2)
    return

def Stacking_NB(train_X, test_X, train_y, train_id, test_id):
    for i in range(5):
        print("part %d begin"%i)
        id_A = train_id[i]
        id_B = test_id[i]
        trainX = train_X[id_A]
        testX = train_X[id_B]
        trainy = train_y[id_A]

        model = MultinomialNB()
        model.fit(trainX, trainy)
        pred = model.predict_proba(testX)
        test_y = model.predict_proba(test_X) 
        with open("Ensemble_Data/Stack_NB_train%d.pkl"%i,'wb') as f1 :
            pickle.dump(pred, f1)
        with open("Ensemble_Data/Stack_NB_test%d.pkl"%i,'wb') as f2:
            pickle.dump(test_y, f2)
    return

def Stacking_xgboost(train_X, test_X, train_y, train_id, test_id):
    print("XGBOOST!!!!!!!")
    for i in range(5):
        print("part %d begin"%i)
        id_A = train_id[i]
        id_B = test_id[i]
        trainX = train_X[id_A]
        testX = train_X[id_B]
        trainy = train_y[id_A]
        testy = train_y[id_B]
        xlf = XGBClassifier(
            max_depth = 9,
            n_estimators = 800,
            min_child_weight = 1,
            scale_pos_weight = 4,
            silent= False,
            objective='binary:logistic',
            nthread=-1)
        xlf.fit(trainX, trainy, eval_metric='auc', verbose = True, eval_set = [(testX, testy)])
        pred = xlf.predict_proba(testX)
        test_y = xlf.predict_proba(test_X)
        with open("Ensemble_Data/Stack_xgboost_train%d.pkl"%i,'wb') as f1 :
            pickle.dump(pred, f1)
        with open("Ensemble_Data/Stack_xgboost_test%d.pkl"%i,'wb') as f2:
            pickle.dump(test_y, f2)
    return

train_X,test_X,train_y = Load_Data(0.7)
X1 = csr_matrix(train_X)
X2 = csr_matrix(test_X)
#train_id, test_id = Split_Data(X1)
train_id, test_id = Load_id()
Stacking_NB(X1, X2, train_y, train_id, test_id)



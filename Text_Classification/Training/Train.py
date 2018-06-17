import json
import time
import re
import jieba
import pickle
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
path = "E:/Study/2017-2018 (Second)/Machine_Learning/_____TASK_____/0/"

def Cut_Word(train_content, train_title,test_content,test_title):
    fn_input = ["train.json","test.json"]
    fn_content = [train_content, test_content]
    fn_title = [train_title, test_title]
    
    file_sw = open(path + "StopWords.txt",'r',encoding = 'UTF-8')
    stop_words = file_sw.read().splitlines()
    for i in range(len(stop_words)):
        stop_words[i] = stop_words[i].strip()
        
    for i in range(2):
        file_content = open(fn_content[i],'wb')
        file_title = open(fn_title[i], 'wb')
        with open(path + fn_input[i], 'r') as f:
            cnt = 0
            for line in f:
                #Read files
                Dict  = json.loads(line)
                soup = BeautifulSoup(Dict['content'],'lxml')
                content = soup.get_text()
                content = jieba.cut_for_search(content)
                title = jieba.cut_for_search(Dict['title'])
                content = list(set(content))
                title = list(set(title))
                
                #delete stop words
                for sw in stop_words:
                    if sw in content:
                        content.remove(sw)
                    if sw in title:
                        title.remove(sw)

                content_copy = content.copy()
                title_copy = title.copy()
                
                for wd in content_copy:
                    if (('0' <= wd[0]) and (wd[0] <= '9')) or wd == '\u3000':
                        content.remove(wd)
                for wd in title_copy:
                    if (('0' <= wd[0]) and (wd[0] <= '9')) or wd == '\u3000':
                        title.remove(wd)
                        
                #save file
                pickle.dump(content, file_content)
                pickle.dump(title, file_title)
                #print status
                if (cnt%1000 == 0):
                    print("Now Processed %d"%cnt)
                cnt += 1
    return

def Get_Cutted_Words():
    f1 = open("train_content.pkl",'rb')
    f2 = open("train_title.pkl",'rb')
    f3 = open("test_content.pkl",'rb')
    f4 = open("test_title.pkl",'rb')
    train_A = []
    train_B = []
    test_A = []
    test_B = []
    for i in range(321910):
        content = pickle.load(f1)
        title = pickle.load(f2)
        train_A.append(" ".join(content))
        train_B.append(" ".join(title))
        
    for i in range(40239):
        content = pickle.load(f3)
        title = pickle.load(f4)
        test_A.append(" ".join(content))
        test_B.append(" ".join(title))
    print("Cutted Word has been loaded!")
    return train_A, train_B, test_A, test_B

def Vectorization():
    train_A, train_B, test_A, test_B = Get_Cutted_Words()
    model_A = TfidfVectorizer(max_features = 100000, ngram_range=(2,2))#min_df = 5e-5)
    model_B = TfidfVectorizer(max_features = 100000, ngram_range=(2,2))#min_df = 5e-5)
    print("begin trainning model A")
    train_XA = model_A.fit_transform(train_A)
    print("Model A fit over")
    test_XA = model_A.transform(test_A)
    print("begin trainning model B")
    train_XB = model_B.fit_transform(train_B)
    print("Model B fit over")
    test_XB = model_B.transform(test_B)

    param = open("param_tfidf.txt",'w')
    param.write(str(model_A.get_feature_names()))
    param.write(str(model_B.get_feature_names()))
    param.close()
                
    file = open('Sparse_Matrix_20w.pkl','wb')
    pickle.dump(train_XA, file)
    pickle.dump(test_XA, file)
    pickle.dump(train_XB, file)
    pickle.dump(test_XB, file)
    file.close()
    print("tfidf over")
    return

def Specific_Vectorization():
    train_A, train_B, test_A, test_B = Get_Cutted_Words()
    train_y = pd.read_csv('train.csv')['pred']
    Series_A = pd.Series(train_A) 
    train_pos = Series_A[train_y == 1]

    model = TfidfVectorizer(max_features = 400000)#min_df = 5e-5)
    model.fit(train_pos)
    train_specific = model.transform(train_A)
    test_specific = model.transform(test_A)
    file = open("Sparse_Matrix_Specific.pkl",'wb')
    pickle.dump(train_specific,file)
    pickle.dump(test_specific,file)
    file.close()
    

def CountVectorization():
    print("begin count vec")
    train_A, train_B, test_A, test_B = Get_Cutted_Words()
    model_A = CountVectorizer(min_df = 1e-3)
    train_X = model_A.fit_transform(train_A)
    test_X = model_A.transform(test_A)
    file = open('Countvec_Matrix.pkl','wb')
    pickle.dump(train_X, file)
    pickle.dump(test_X, file)
    file.close()
    print("count vec over")
    print(train_X.shape)
    return train_X, test_X
    

def Load_Data(times):
    file = open('Sparse_Matrix_80w.pkl','rb')
    train_XA = pickle.load(file)
    test_XA = pickle.load(file)
    train_XB = pickle.load(file)
    test_XB = pickle.load(file)
    file.close()
    train_X = sparse.hstack([train_XA, train_XB*times])
    test_X = sparse.hstack([test_XA, test_XB*times])
    print("Load Data Complete!")
    print(train_X.shape)
    return train_X, test_X


def Train(train_X, test_X, train_y,c):
    print("LR training:C = %f"%c)
    model = LogisticRegression(C = c, penalty='l1')
    model.fit(train_X,train_y)
    pred = model.predict_proba(test_X)
    train_pred = model.predict_proba(train_X)
    return pred,train_pred

def NBC(train_X, test_X, train_y):
    #model = MultinomialNB()
    model = BernoulliNB()
    model.fit(train_X,train_y)
    pred = model.predict_proba(test_X)
    return pred

def XGBoost_Train_Test_Split(train_X, test_X, train_y,mcw):
    #X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.15, random_state=1351)
    train_X = csr_matrix(train_X)
    test_X = csr_matrix(test_X)
    num = len(train_y)
    X_train = train_X[:int(num*0.8),:]
    X_test = train_X[int(num*0.8):,:]
    y_train = train_y[:int(num*0.8)]
    y_test = train_y[int(num*0.8):]
    xlf = XGBClassifier(
        max_depth = 9,
        #max_leaves = 40 
        n_estimators = 400,
        min_child_weight = mcw,
        scale_pos_weight = 4,
        silent= False,
        objective='binary:logistic',
        nthread=-1)
    xlf.fit(X_train, y_train, eval_metric='auc', verbose = True, eval_set = [(X_test, y_test)])
    #xlf.fit(train_X, train_y, eval_metric='auc', verbose = True)
    pred = xlf.predict_proba(test_X)
    auc = roc_auc_score(train_y,pred[:1])
    return pred, auc

def XGBoost(train_X, test_X, train_y):
    xgb_train = xgb.DMatrix(train_X, train_y)
    xgb_test = xgb.DMatrix(test_X)
    params = {
        'max_depth' : 9,
        'min_child_weight' : 1,
        #'max_leaf_nodes' = 40,
        #'scale_pos_weight' : 4,
        'silent': False,
        'objective':'binary:logistic',
        'nthread': -1,
        'eval_metric':'auc'
        }
    #xlf.fit(train_X, train_y, eval_metric='auc', verbose = True, eval_set = [(train_X, train_y)])
    #pred = xlf.predict_proba(test_X)
    watchlist=[(xgb_train,'train')]
    num_round = 1000
    model = xgb.train(params, xgb_train,num_round,watchlist,early_stopping_rounds = 100)
    file = open('Model.pkl','wb')
    pickle.dump(model,file)
    file.close()
    pred = model.predict(xgb_test)
    print(pred)
    return pred

def print_best_score(gsearch,param_test):  
     # 输出best score  
    print("Best score: %0.3f" % gsearch.best_score_)  
    print("Best parameters set:")  
    # 输出最佳的分类器到底使用了怎样的参数  
    best_parameters = gsearch.best_estimator_.get_params()  
    for param_name in sorted(param_test.keys()):  
        print("\t%s: %r" % (param_name, best_parameters[param_name])) 

def CV_xgboost(train_X, train_y):
    param_test ={
        'n_estimators':[500,300,140]
        }
    estimator = XGBClassifier(
        max_depth = 9,
        n_estimators = 1000,
        min_child_weight = 1,
        scale_pos_weight = 4,
        silent= False,
        objective='binary:logistic',
        nthread=-1)
    gsearch = GridSearchCV(
        estimator = estimator,
        param_grid = param_test,
        scoring='roc_auc',
        verbose = 10,
        cv = 5) 
    gsearch.fit(train_X, train_y)

    gsearch.grid_scores_, gsearch.best_params_,     gsearch.best_score_
    #print_best_score(gsearch, param_test)
    cv_result = pd.DataFrame.from_dict(gsearch.cv_results_)
    with open('cv_result2.csv','w') as f:
        cv_result.to_csv(f)

    return gsearch

    
def Write_Ans(pred):
    sample = pd.read_csv("sample_submission.csv")
    ID = sample['id']
    DF = pd.DataFrame({'id':ID,'pred':pred})
    DF.to_csv("Submission.csv",index = False)
    return

#Cut_Word("train_content.pkl","train_title.pkl","test_content.pkl","test_title.pkl")
Vectorization()
#Specific_Vectorization()
'''
train_X, test_X = Load_Data(0.7)
train_y = pd.read_csv('train.csv')['pred']

xlf = XGBClassifier(
        max_depth = 10,
        n_estimators = 1200,
        min_child_weight = 1,
        scale_pos_weight = 4,
        silent= False,
        objective='binary:logistic',
        nthread=-1)
xlf.fit(train_X, train_y, eval_metric='auc', verbose = True, eval_set = [(train_X, train_y)])
#xlf.fit(train_X, train_y, eval_metric='auc', verbose = True)
pred = xlf.predict_proba(test_X)
file = open('Model.pkl','wb')
pickle.dump(xlf,file)
file.close()
Write_Ans(pred[:,1])
'''








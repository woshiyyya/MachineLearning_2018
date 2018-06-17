# coding= utf-8
import numpy as np
import jieba
import json
import pickle
import re
from scipy import sparse
import pandas
import jieba.posseg as psg
import jieba.analyse
from xgboost import XGBClassifier
from bs4 import BeautifulSoup
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
path = "E:/Study/2017-2018 (Second)/Machine_Learning/_____TASK_____/0/"
data_path = "E:/Study/2017-2018 (Second)/Machine_Learning/_____TASK_____/0/Feature_Engineering/data"
sparse_path ="E:/Study/2017-2018 (Second)/Machine_Learning/_____TASK_____/0/Reconstruct"
train_num = 321910
test_num = 40239

def Text_Rank():
    fn_output = ["train_TR.pkl", "test_TR.pkl"]
    fn_input = ["train.json", "test.json"]
    for i in range(2):
        out = open(data_path +fn_output[i],'wb')
        with open(path + fn_input[i], 'r') as f:
            cnt = 0
            for line in f:
                cnt += 1
                Dict  = json.loads(line)
                soup = BeautifulSoup(Dict['content'],'lxml')
                S = soup.get_text()
                S = jieba.analyse.textrank(sentence = S, topK = 50, allowPOS = ('ns','n','nr','nt','vn','v'))
                S = " ".join(S)
                pickle.dump(S, out)
                if(cnt % 1000 == 0):
                    print(cnt)
        out.close()

def Text_Length():
    fn_input = ["train.json", "test.json"]
    case_num = [train_num, test_num]
    file = open("Sentence_length.pkl",'wb')
    for i in range(2):
        WN = np.zeros((case_num[i], 12))
        Text_NUM = []
        cnt = 0
        with open(path + fn_input[i], 'r') as f:
            for line in f:
                Dict  = json.loads(line)
                soup = BeautifulSoup(Dict['content'],'lxml')
                S = soup.get_text()
                Text_NUM.append(len(S))
                List = re.split('[，。？、：！\n\t\r]| ',S)
                for l in List:
                    if(len(l) < 55):
                        WN[cnt, int(len(l)/5)] += 1
                    else:
                        WN[11] += 1
                WN[cnt,:] /= len(List)
                if(cnt % 1000 == 0):
                    print(cnt)
                    print(WN[cnt,:])
                cnt += 1
        pickle.dump(WN, file)
        pickle.dump(Text_NUM, file)
    file.close()
    return 


def Word_Property():
    fn_input = ["train.json", "test.json"]
    out = open('data/WP_Matrix.pkl','wb')
    Num = [train_num, test_num]
    Map = {
        'a':1,'ad':2,'an':3,'ag':4,'al':5,'b':6,'bl':7,'c':8,'cc':9,'d':10,'e':11,'f':12,'h':13,'k':14,'m':15,'mq':16,'n':17,'nr':18,'nr1':19,'nr2':20,
        'nrj':21,'nrf':22,'ns':23,'nsf':24,'nt':25,'nz':26,'nl':27,'ng':28,'o':29,'p':30,'pba':31,'pbei':32,'q':33,'qv':34,'qt':35,'r':36,'rr':37,
        'rz':38,'rzt':39,'rzs':40,'rzv':41,'ry':42,'ryt':43,'rys':44,'ryv':45,'rg':46,'s':47,'t':48,'tg':49,'u':50,'uzhe':51,'ule':52,'uguo':53,'ude1':54,
        'ude2':55,'ude3':56,'usuo':57,'udeng':58,'uyy':59,'udh':60,'uls':61,'uzhi':62,'ulian':63,'v':64,'vd':65,'vn':66,'vshi':67,'vyou':68,
        'vf':69,'vx':70,'vi':71,'vl':72,'vg':73,'w':74,'wkz':75,'wky':76,'wyz':77,'wyy':78,'wj':79,'ww':80,'wt':81,'wd':82,'wf':83,'wn':84,'wm':85,'ws':86,
        'wp':87,'wb':88,'wh':89,'x':90,'xx':91,'xu':92,'y':93,'z':0
    }
    Map_size = 94

    for i in range(2):
        Count_Mat = np.zeros((Num[i], 200))
        with open(path + fn_input[i], 'r') as f:
            cnt = 0
            for line in f:
                Dict  = json.loads(line)
                soup = BeautifulSoup(Dict['content'],'lxml')
                X = psg.cut(soup.get_text())
                wd_num = 0
                for x in X:
                    wd_num += 1
                    if x.flag in Map:
                        Count_Mat[cnt, Map[x.flag]] += 1
                    else: 
                        Map[x.flag] = Map_size
                        Map_size += 1
                        Count_Mat[cnt, Map[x.flag]] += 1
                Count_Mat[cnt,:] /= wd_num
                if(cnt % 1000 == 0):
                    print(cnt)
                cnt += 1
        sparse_mat =  csr_matrix(Count_Mat[:,0:Map_size])
        pickle.dump(sparse_mat, out)    
    out.close()

def Vectorization():
    train = open("data/train_TR.pkl",'rb')
    test = open("data/test_TR.pkl",'rb')
    print("LOADING......")
    train_text = []
    test_text = []
    for _ in range(train_num):
        train_text.append(pickle.load(train))
    for _ in range(test_num):
        test_text.append(pickle.load(test))
    print("Load Complete!")
    text = train_text + test_text
    model = CountVectorizer()
    model.fit(text)
    train_X = model.transform(train_text)
    test_X = model.transform(test_text)
    print("Vectorization Complete! With dimension: ", train_X.shape)

    file = open("data/Text_Rank_Matrix.pkl",'wb')
    pickle.dump(train_X,file)
    pickle.dump(test_X, file)
    file.close()
    return train_X, test_X

def Write_Ans(pred):
    sample = pandas.read_csv("data/sample_submission.csv")
    ID = sample['id']
    DF = pandas.DataFrame({'id':ID,'pred':pred})
    DF.to_csv("Submission.csv",index = False)
    return

def Text_URL():
    path = "E:/Study/2017-2018 (Second)/Machine_Learning/_____TASK_____/0/Feature_Engineering/URLs/"
    train_file = open(path + "train_urls_new.pkl", 'rb')
    test_file = open(path + "test_urls_new.pkl", 'rb')
    train_url = pickle.load(train_file)
    test_url = pickle.load(test_file)
    train = []
    test = []
    for List in train_url:
        train.append(" ".join(List))
    for List in test_url:
        test.append(" ".join(List))
    model = TfidfVectorizer()
    train_X = model.fit_transform(train)
    test_X = model.transform(test)
    with open("data/Sparse_Matrix_url.pkl", 'wb') as f:
        pickle.dump(train_X, f)
        pickle.dump(test_X, f)
    print(train_X.shape)
    print(test_X.shape)
    return train_X, test_X


def Load_Data():
    TR = open("data/Text_Rank_Matrix.pkl",'rb')
    WP = open('data/WP_Matrix.pkl','rb')
    SP = open(sparse_path+"/Sparse_Matrix_25w.pkl",'rb')
    NG = open(sparse_path+"/Sparse_Matrix_20w.pkl",'rb')
    WN = open('data/Sentence_length.pkl','rb')
    TU = open('data/Sparse_Matrix_url.pkl','rb')
    train_y = pandas.read_csv("data/train.csv")['pred']
    train_XA = pickle.load(SP)
    test_XA = pickle.load(SP)
    train_XB = pickle.load(SP)
    test_XB = pickle.load(SP)
    train_XA1 = pickle.load(NG)
    test_XA1 = pickle.load(NG)
    train_XB1 = pickle.load(NG)
    test_XB1 = pickle.load(NG)
    #train_XC, test_XC = Vectorization()
    train_XC = pickle.load(TR)
    test_XC = pickle.load(TR)
    train_XD = pickle.load(WP)
    test_XD = pickle.load(WP)
    train_XE = csr_matrix(pickle.load(WN))
    train_XF = csr_matrix(np.array(pickle.load(WN)).reshape(train_num,1))
    test_XE = csr_matrix(pickle.load(WN))
    test_XF = csr_matrix(np.array(pickle.load(WN)).reshape(test_num,1))
    train_XG = pickle.load(TU) 
    test_XG = pickle.load(TU)
    train_X = sparse.hstack([train_XA, train_XB,train_XA1, train_XB1,train_XC, train_XD,train_XE, train_XF, train_XG])
    test_X = sparse.hstack([test_XA, test_XB,test_XA1, test_XB1,test_XC, test_XD,test_XE, test_XF,test_XG])
    #train_X = sparse.hstack([train_XA, train_XB,train_XC, train_XD,train_XE, train_XF])
    #test_X = sparse.hstack([test_XA, test_XB,test_XC, test_XD,test_XE, test_XF])
    
    print("Load Data Complete!", train_X.shape)
    return train_X, test_X, train_y

def Load_model(file_name):
    file = open(file_name,'rb')
    model = pickle.load(file)
    file.close()
    return model



train_X, test_X, train_y = Load_Data()
'''
#------------------train Parameters--------------------
X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.2, random_state=420)
xlf = XGBClassifier(
        eta = 0.05, 
        max_depth = 13,
        n_estimators = 1200,
        min_child_weight = 2,
        #scale_pos_weight = 4,
        silent= False,
        objective='binary:logistic',
        #subsample=1,
        nthread=-1
        )
xlf.fit(X_train, y_train, eval_metric='auc', verbose = True, eval_set = [(X_test, y_test)], early_stopping_rounds=15)

file = open('Model_10_url.pkl','wb')
model = pickle.dump(xlf,file)
file.close()


#-------------------train full data---------------------
xlf = XGBClassifier(
        eta = 0.1,
        max_depth = 13,
        n_estimators = 1000,
        min_child_weight = 2,
        scale_pos_weight = 4,
        silent= False,
        objective='binary:logistic',
        #subsample [default=1]
        nthread=-1
        )
xlf.fit(train_X, train_y, eval_metric='auc', verbose = True, eval_set = [(train_X,train_y)])
#pred = xlf.predict_proba(test_X)

file = open('Model_13_full_url.pkl','wb')
model = pickle.dump(xlf, file)
file.close()

'''
model = Load_model('Model_13_full_url.pkl')
pred = model.predict_proba(test_X,ntree_limit = 1000)
Write_Ans(pred[:,1])


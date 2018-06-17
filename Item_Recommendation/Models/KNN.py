from scipy.sparse import lil_matrix
import pickle
import pandas
from sklearn.cluster import KMeans
import numpy as np
from scipy.sparse import coo_matrix
from xgboost import XGBRegressor

def preprocess():
    names = ["id","f1","f2","af1","af2","af3","af4"]
    Feature = pandas.read_csv("data/release/feature.txt", names = names, sep='\t')
    location = [0, 51765, 71350, 71350, 71350, 71350]
    total_num = Feature.shape[0]
    feature_mat = lil_matrix((total_num, 71556))

    for i in range(total_num):
            if(i%10000 == 0):
                print(i)
            case = ""
            for j in range(1,7):
                loc = location[j - 1]
                val = Feature.ix[i, names[j]]
                if(val == -1):
                    continue
                feature_mat[i, loc + val] = 1

    out = open("data/KNN/Feature_Mat.pkl",'wb')
    pickle.dump(feature_mat , out)
    out.close()

def construct_SP():
    day_base =  0
    time_base = 6664
    user_base = 6664 + 5178
    item_base = 6664 + 5178 + 60928
    total_base = item_base + 497276
    Name = ["user_id", "item_id", "score", "day", "time"]
    train = pandas.read_csv("data/release/train.tsv", names = Name, sep='\t')
    valid = pandas.read_csv("data/release/valid.tsv", names = Name, sep='\t')
    train = pandas.concat([train,valid],ignore_index=True)
    row = []
    col = []
    train_base = train.shape[0]
    for i in range(train.shape[0]):
        row.append(i)
        row.append(i)
        row.append(i)
        row.append(i)
        col.append(day_base + train.ix[i, 'day'])
        col.append(time_base + train.ix[i, 'time'])
        col.append(user_base + train.ix[i, 'user_id'])
        col.append(item_base + train.ix[i, 'item_id'])
        if(i % 1000 == 0):
            print(i)

    out = open("data/KNN/Full.pkl",'wb')
    pickle.dump(row , out)
    pickle.dump(col , out)
    out.close()

def construct_SPV():
    day_base =  0
    time_base = 6664
    user_base = 6664 + 5178
    item_base = 6664 + 5178 + 60928
    total_base = item_base + 497276
    Name = ["user_id", "item_id", "score", "day", "time"]
    train = pandas.read_csv("data/release/valid.tsv", names = Name, sep='\t')
    row = []
    col = []
    for i in range(train.shape[0]):
        row.append(i)
        row.append(i)
        row.append(i)
        row.append(i)
        col.append(day_base + train.ix[i, 'day'])
        col.append(time_base + train.ix[i, 'time'])
        col.append(user_base + train.ix[i, 'user_id'])
        col.append(item_base + train.ix[i, 'item_id'])
        if(i % 1000 == 0):
            print(i)

    out = open("data/KNN/Valid.pkl",'wb')
    pickle.dump(row , out)
    pickle.dump(col , out)
    out.close()

def Sparse():
    In = open("data/KNN/Valid.pkl",'rb')
    row = pickle.load(In)
    print("load row ok")
    col = pickle.load(In)
    print("load col ok")
    In.close()
    One = np.ones(len(row))
    Data = coo_matrix((One, (row, col)))
    print("coo matrix ok")
    Out = open("data/KNN/coo_matrix_v.pkl",'wb')
    pickle.dump(Data, Out)
    Out.close()



def KNN():
    IN = open("data/KNN/coo_matrix.pkl",'rb')
    data = pickle.load(IN)  
    IN.close()
    print("begin trainning..............")
    model = KMeans(n_clusters = 100,n_init = 1,verbose=True)
    model.fit(data)
    ANS = model.predict(data)
    pandas.DataFrame(ANS).to_csv("data/KNN/pred.csv")


def XGB():
    IN = open("data/KNN/coo_matrix.pkl",'rb')
    X_train = pickle.load(IN)  
    IN.close()

    IN = open("data/KNN/coo_matrix_v.pkl",'rb')
    X_valid = pickle.load(IN)  
    IN.close()

    Name = ["user_id", "item_id", "score", "day", "time"]
    train = pandas.read_csv("data/release/train.tsv", names = Name, sep='\t')
    valid = pandas.read_csv("data/release/valid.tsv", names = Name, sep='\t')
    train = pandas.concat([train,valid],ignore_index=True)
    y_train = train['score']
    y_valid = valid['score']
    print("trainnnnnnnniiingggg")
    xlf = XGBRegressor(
        max_depth = 4,
        #max_leaves = 40 
        n_estimators = 50,
        silent= False,
        nthread=-1)
    xlf.fit(X_train, y_train, eval_metric='rmse', verbose = True, eval_set = [(X_valid, y_valid)])

XGB()
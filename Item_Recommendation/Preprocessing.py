import pandas
import pickle
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
from scipy import sparse
from sklearn.cluster import KMeans
from surprise import SVDpp


data_path = "E:\\Study\\2017-2018 (Second)\\Machine_Learning\\_____TASK_____\\task2\\data\\release\\"
names = ["id","f1","f2","af1","af2","af3","af4"]
Train = pandas.read_csv(data_path + "train.tsv", names = ['user','item','rating','day','time'], sep="\t")
Feature = pandas.read_csv(data_path + "feature.txt", names = names, sep="\t")
Feature_onehot_path = "data\\transform\\Feature_onehot.pkl"
user_num = 60927
item_num = 497658
case_num = 11976538

def Save_Pickle(data, filename):
    file = open("data\\transform\\" + filename, "wb")
    for d in data:
        pickle.dump(d, file)
    file.close()


def OneHot():
    Feature_onehot_file = open(Feature_onehot_path, 'wb')
    Item_num = len(Feature['id'])
    F1_dim = max(Feature['f1']) + 1
    F2_dim = max(Feature['f2']) + 1
    AF_dim = max(Feature['af1']) + 1
    total_dim = F1_dim + F2_dim + AF_dim
    Feature_onehot = lil_matrix((Item_num, total_dim))
    print("begin process feature! " + str(Feature_onehot.shape))

    for i in range(Item_num):
        Row = Feature.iloc[i]
        Feature_onehot[i, Row['f1']] = 1
        Feature_onehot[i, F1_dim + Row['f2']] = 1
        for j in [3,4,5,6]:
            if(Row[names[j]] == -1):
                break
            Feature_onehot[i, F1_dim + F2_dim + Row[names[j]]] = 1
        if(i%1000 == 0):
            print(i)

    pickle.dump(Feature_onehot, Feature_onehot_file)
    Feature_onehot_file.close()


def Transform():
    Feature_onehot_file = open(Feature_onehot_path, 'rb')
    Feature = pickle.load(Feature_onehot_file)

    feature_num = Feature.shape[1]
    User_mat = lil_matrix((case_num, user_num + 2))
    Feature_mat = lil_matrix((case_num, feature_num))
    for i in range(case_num):
        Row = Train.iloc[i]
        User_mat[i, Row['user']] = 1
        User_mat[i, user_num] = Row['day']
        User_mat[i, user_num + 1] = Row['time']
        Feature_mat[i,:] = Feature[Row['item']]
        if(i % 1000 == 0):
            print(i)
    Train_Mat = sparse.hstack([User_mat, Feature_mat])
    print("Complete!!" + str(Train_Mat.shape))
    Train_file = open("data\\transform\\Train.pkl", "wb")
    pickle.dump(Train_Mat, Train_file)
    pickle.dump(Train["rating"], Train_file)
    Train_file.close()

def KNN():
    Feature_onehot_file = open(Feature_onehot_path, 'rb')
    Feature = pickle.load(Feature_onehot_file)
    clf = KMeans(n_clusters = 20, verbose = 1)   
    clf.fit(Feature)
    Cluster = clf.predict(Feature)
    file = open("data\\transform\\KMeans.pkl", "wb")
    pickle.dump(clf, file)
    pickle.dump(Cluster, file)
    file.close()

def UIMatrix():
    UIMatrix = lil_matrix((user_num, item_num))
    print(UIMatrix.shape)
    for i in range(case_num):
        Row = Train.iloc[i]
        UIMatrix[Row['user'], Row['item']] = Row['rating']
        if(i % 10000 == 0):
            print(i)
    Save_Pickle([UIMatrix], "UIMatrix2.pkl")

def SVDPP():
    file = open("data\\transform\\UIMAtrix.pkl", "rb")
    UIMatrix = pickle.load(file)
    print(UIMatrix[2,3])
    Mat = csr_matrix(UIMatrix)
    print(Mat.shape)
    file.close()
    #model = SVDpp(n_factors = 30)
    #model.fit(UIMatrix)

UIMatrix()





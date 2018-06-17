import pandas 
import pickle
from sklearn.metrics import mean_squared_error as mse
from math import sqrt
Day = 6665
Time = 5179
User = 60928
Item = 497657
F1 = 51765
F2 = 19585
AF = 206

feature_name = ["id","f1","f2","af1","af2","af3","af4"]
test_name = ["user_id", "item_id", "day", "time"]
train_name = ["user_id", "item_id", "score", "day", "time"]


def peekdata():
    test_file = "data/release/test.tsv"
    Test = pandas.read_csv(test_file, names = test_name, sep='\t')
    print(max(Test['day']))
    print(max(Test['time']))
    print(max(Test['user_id']))
    print(max(Test['item_id']))

def peekpred():
    test_file = "data/SVDF_Data/ensemble/submission1.csv"
    Test = pandas.read_csv(test_file)
    print(max(Test['pred']))
    print(min(Test['pred']))


def handle_feature():
    feature_file = "data/release/feature.txt"
    train_file = "data/release/train.tsv"
    test_file = "data/release/test.tsv"
    Feature = pandas.read_csv(feature_file, names = feature_name, sep='\t')
    Train = pandas.read_csv(train_file, names = train_name, sep='\t')
    Test = pandas.read_csv(test_file, names = test_name, sep='\t')
    
    feature_out = open("data/SVDF_Data/feature.pkl", "wb")
    Train_out = open("data/SVDF_Data/train.txt", "w")
    Test_out = open("data/SVDF_Data/test.txt", "w")
    feature = []
    feature_num = []
    B1 = Item
    B2 = Item + F1
    B3 = Item + F1 + F2
    Base = [0, B1, B2, B3, B3, B3, B3]
    for i in range(Feature.shape[0]):
        case = ""
        cnt = 0
        for j in range(7):
            offset = Feature.ix[i, feature_name[j]]
            if(offset != -1):
                cnt = cnt + 1 
                case = case + str(Base[j] + offset) + ":1 "
        if(i % 10000 == 0):
            print(str(i) + "   " + case) 
        feature.append(case)
        feature_num.append(cnt) 
    
    pickle.dump(feature, feature_out)
    pickle.dump(feature_num, feature_out)
    feature_out.close()
    print("Feature complete!")


    valid_file = "data/release/valid.tsv"
    Valid_out = open("data/SVDF_Data/valid.txt", "w")
    Valid = pandas.read_csv(valid_file, names = train_name, sep='\t')
    for i in range(Valid.shape[0]):
        item_id = Valid.ix[i, "item_id"]
        case = ""
        case = case + str(Valid.ix[i, "score"]) 
        case = case + " 2 1 " + str(feature_num[item_id]) + " "
        case = case + str(Valid.ix[i, "day"]) + ":1 "
        case = case + str(Day + Valid.ix[i, "time"]) + ":1 "
        case = case + str(Valid.ix[i, "user_id"]) + ":1 "
        case = case + feature[item_id] + "\n"
        Valid_out.writelines(case)
        if(i % 100000 == 0):
            print(str(i) + "   " + case)
    
    Valid_out.close()
    print("Valid Complete!")
    '''
    for i in range(Train.shape[0]):
        item_id = Train.ix[i, "item_id"]
        case = ""
        case = case + str(Train.ix[i, "score"]) 
        case = case + " 2 1 " + str(feature_num[item_id]) + " "
        case = case + str(Train.ix[i, "day"]) + ":1 "
        case = case + str(Day + Train.ix[i, "time"]) + ":1 "
        case = case + str(Train.ix[i, "user_id"]) + ":1 "
        case = case + feature[item_id] + "\n"
        Train_out.writelines(case)
        if(i % 100000 == 0):
            print(str(i) + "   " + case)
    
    Train_out.close()
    print("Train Complete!")

    for i in range(Test.shape[0]):
        item_id = Test.ix[i, "item_id"]
        case = "0"
        case = case + " 2 1 " + str(feature_num[item_id]) + " "
        case = case + str(Test.ix[i, "day"]) + ":1 "
        case = case + str(Test.ix[i, "time"] + Day) + ":1 "
        case = case + str(Test.ix[i, "user_id"]) + ":1 "
        case = case + feature[item_id] + "\n"
        Test_out.writelines(case)
        if(i % 100000 == 0):
            print(str(i) + "   " + case)
    
    Test_out.close()
    print("Test Complete!")
'''

def Normalize():
    valid_file = "data/release/valid.tsv"
    pred = pandas.read_csv("data/SVDF_Data/normalize/pred.txt",names=["pred"])
    Valid = pandas.read_csv(valid_file, names = train_name, sep='\t')
    ANS = Valid['score']
    MAX = max(pred['pred'])
    MIN = min(pred['pred'])
    Range = MAX - MIN
    PRED = (pred['pred'] - MIN)*100/Range
    print(max(PRED))
    print(max(pred['pred']))
    print(max(ANS))
    print(sqrt(mse(ANS, pred['pred'])))
    print(sqrt(mse(ANS, PRED)))
    print(ANS.head)
    print(PRED.head)
    print(pred['pred'].head)

def Write_Ans(fin, fout):
    Ans = pandas.read_csv("E:/Study/2017-2018 (Second)/Machine_Learning/_____TASK_____/task2/data/release/random.csv")
    pred = []
    i = 0
    with open(fin, 'r') as f:
        for a in f.readlines():
            pred.append(float(a.strip("\n")))
            i = i + 1
            if(i % 1000 == 0):
                print(i)
    Ans['pred'] = pandas.Series(pred)
    Ans.to_csv(fout, index = False)
    return

#handle_feature()        
#peekdata()
#peekpred()
#Normalize()
Write_Ans("data/SVDF_Data/pred.txt","data/SVDF_Data/out/submission.csv")
#Write_Ans("data/SVDF_Data/pred.txt", "data/SVDF_Data/submission.csv")
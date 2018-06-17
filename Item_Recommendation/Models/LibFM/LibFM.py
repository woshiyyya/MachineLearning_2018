import pandas 
import pickle
names = ["id","f1","f2","af1","af2","af3","af4"]
Name = [["user_id", "item_id", "day", "time"],
        ["user_id", "item_id", "score", "day", "time"]]
#location = [0, 497657, 549422, 569007, 569007, 569007, 569007]
#f1 f2 af1 af2 af3 af4
location = [0, 51765, 71350, 71350, 71350, 71350, 71555]
base = 60504 + 497657
day_base = 71555
time_base = 71555 + 6665
user_num = 60504
#for i in range(6):
#    location[i] = location[i] + base

Day = 6665
Time = 5178
User = 60927
Item = 497276

print(location)

# Data Format: 
# score <user_id> <item_id> <feature1> <feature2> <add_feature> <day> <time>

def item_feature(fin, fout):
    item_feature_matrix = []
    Feature = pandas.read_csv(fin, names = names, sep='\t')
    KNN = pandas.read_csv("data/KNN/pred.csv")
    Feature['knn'] = KNN['0']
    total_num = Feature.shape[0]
    print("total item num:", total_num)
    for n in names:
        print(n,": ",max(Feature[n]) + 1)

    for i in range(total_num):
        if(i%10000 == 0):
            print(i)
        case = ""
        for j in range(1,8):
            loc = location[j - 1]
            val = Feature.ix[i, names[j]]
            if(val == -1):
                continue
            case = case + str(loc + val) + ":1 " 
        item_feature_matrix.append(case)
        
    print(len(item_feature_matrix))
    out = open(fout, 'wb')
    pickle.dump(item_feature_matrix, out)
    out.close()
    return

def Construct_Matrix(fin, fout, feature_file, iftrain):
    with open(feature_file, 'rb') as f:
        feature = pickle.load(f)
    Out = open(fout, 'w')

    day_base = 71555
    time_base = 71555 + 12
    user_base = 71555 + 12 + 24
    item_base = 71555 + 12 + 24 + 60928
    names = Name[iftrain]
    Data = pandas.read_csv(fin, names = names, sep = '\t')
    total_num = Get_info(Data)

    for i in range(total_num):
        case = ""
        if(iftrain):
            case = case + str(Data.ix[i, 'score']) + " "
        else:
            case = case + "0 "

        case = case + feature[Data.ix[i, 'item_id']]
        case = case + str(day_base + int(Data.ix[i, 'day']/555.3)) + ":1 "
        case = case + str(time_base + int(Data.ix[i, 'time']/215.75)) + ":1 "
        case = case + str(user_base + Data.ix[i, 'user_id']) + ":1 "
        case = case + str(item_base + Data.ix[i, 'item_id']) + ":1 "
        case += "\n"
        if(i % 1000 == 0):
            print(i)
        Out.writelines(case)
    Out.close()
    return

def group_gen():
    day_base = 71555 + 2000
    time_base = 71555 + 6664 + 2000
    user_base = 71555 + 6664 + 5178 + 2000
    item_base = 71555 + 6664 + 5178 + 60928 + 2000
    total_base = item_base + 497276
    f = open("meta.txt",'w')
    for i in range(71555):
        f.writelines("0\n")
    for i in range(2000):
        f.writelines("1\n")
    for i in range(6664):
        f.writelines("2\n")
    for i in range(5178):
        f.writelines("3\n")
    for i in range(60928):
        f.writelines("4\n")
    for i in range(497276):
        f.writelines("5\n")
    f.close()


def Construct_Matrix_drop(fin, fout, feature_file, iftrain):
    with open(feature_file, 'rb') as f:
        feature = pickle.load(f)
    Out = open(fout, 'w')

    names = Name[iftrain]
    Data = pandas.read_csv(fin, names = names, sep = '\t')
    total_num = Get_info(Data)

    for i in range(total_num):
        case = ""
        if(iftrain):
            case = case + str(Data.ix[i, 'score']) + " "
        else:
            case = case + "0 "
        case = case + feature[Data.ix[i, 'item_id']]
        case = case + str(Data.ix[i, 'user_id']) + ":1 "
        case = case + str(Data.ix[i, 'item_id'] + user_num) + ":1 "
        case = case + str(day_base + Data.ix[i, 'day']) + ":1 "
        case = case + str(time_base + Data.ix[i, 'time']) + ":1 "
        case += "\n"
        if(i % 100000 == 0):
            print(i + "   " + case)
        Out.writelines(case)
    Out.close()
    return

def Get_info(Data):
    total_num = Data.shape[0]
    print("total_num: ", total_num)
    return total_num

def Peek_Data():
    Data = pandas.read_csv("data/release/train.tsv", names = Name[1], sep = '\t')
    print(max(Data['day']))
    print(max(Data['time']))
    print(max(Data['user_id']))
    print(max(Data['item_id']))

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

def simple_matrix(fin, fout, iftrain, prefix):
    out = open(fout, 'w')
    names = Name[iftrain]
    Data = pandas.read_csv(fin, names = names, sep = '\t')
    offset = max(Data['user_id']) + 1
    total_num = Get_info(Data)
    for i in range(total_num):
        case = prefix + str(Data.ix[i, 'user_id']) + ":1 " + str(Data.ix[i, 'item_id'] + offset) + ":1 "  + "\n"
        out.writelines(case)
        if(i % 1000 == 0):
            print(i)
    out.close()
    return

def Append(fin1, fin2, fout):
    f1 = open(fin1, 'r')
    f2 = open(fin2, 'r')
    fo = open(fout, 'w')
    for i in f1.readlines():
        fo.writelines(i)
    f1.close()
    for i in f2.readlines():
        fo.writelines(i)
    f2.close()
    fo.close()
'''
item_feature("data/release/feature.txt","data/out/feature_mat_knn.pkl") 
'''
Construct_Matrix("data/release/test.tsv","data/out/test_mat.txt",
        "data/out/feature_mat3.pkl", 0)
Construct_Matrix("data/release/valid.tsv","data/out/valid_mat.txt",
        "data/out/feature_mat3.pkl", 1)
Construct_Matrix("data/release/train.tsv","data/out/train_mat.txt",
        "data/out/feature_mat3.pkl", 1)
Append("data/out/train_mat.txt","data/out/valid_mat.txt","data/out/Full_mat.txt")

#Construct_Matrix("data/release/valid.tsv","data/out/valid_mat.txt",
#        "data/out/feature_mat.pkl", 1)
#simple_matrix("data/release/test.tsv", "data/out/simpletest_mat.txt",0 , "0 ")

#Peek_Data()

#Write_Ans("data/SVDF_Data/pred.txt","data/SVDF_Data/out/submission.csv")

group_gen()
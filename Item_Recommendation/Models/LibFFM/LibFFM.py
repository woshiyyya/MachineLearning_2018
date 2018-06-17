import pandas
import pickle

location = [0, 51765, 71350, 71350, 71350, 71350, 71555]
names = ["id","f1","f2","af1","af2","af3","af4"]

def item_feature():
    item_feature_matrix = []
    Feature = pandas.read_csv("data/release/feature.txt", names = names, sep='\t')
    out = open("LIBFFM/Feature_mat.pkl", 'wb')

    LIST = []
    for i in range(Feature.shape[0]):
        if(i%10000 == 0):
            print(i)
        case = ""
        f1 = Feature.ix[i, 'f1']
        f2 = Feature.ix[i, 'f2']
        af = [Feature.ix[i, 'af1'],Feature.ix[i, 'af2'],Feature.ix[i, 'af3'],Feature.ix[i, 'af4']]
       
        if(f1 != -1):
            case = case + "0:" + str(f1) + ":1 " 
        if(f2 != -1):
            case = case + "1:" + str(f2) + ":1 "
        for x in af:
            if(x != -1):
                case = case + "2:" + str(x) + ":1 "
        LIST.append(case)    
    pickle.dump(LIST, out)
    out.close()
    return

def Construct_Mat():
    names = ["user_id", "item_id", "score", "day", "time"]
    names_test = ["user_id", "item_id", "day", "time"]
    feature_f = open("LIBFFM/Feature_mat.pkl", 'rb')
    train = pandas.read_csv("data/release/train.tsv", names = names, sep = '\t')
    valid = pandas.read_csv("data/release/valid.tsv", names = names, sep = '\t')
    test = pandas.read_csv("data/release/test.tsv", names = names_test, sep = '\t')
    features = pickle.load(feature_f)
    feature_f.close()
    out1 = open("LIBFFM/full.txt",'a')
    out2 = open("LIBFFM/test.txt",'w')
 
    for i in range(valid.shape[0]):
        case = ""
        case = case + str(valid.ix[i, 'score'])
        case = case + features[valid.ix[i, "item_id"]]
        case = case + "3:" + str(valid.ix[i, "user_id"]) + ":1 "
        case = case + "4:" + str(valid.ix[i, "item_id"]) + ":1 "
        case = case + "5:" + str(valid.ix[i, "day"]) + ":1 "
        case = case + "6:" + str(valid.ix[i, "time"]) + ":1 "
        case = case + "\n"
        out1.writelines(case)
        if(i % 1000 == 0):
            print(i)

    out1.close()

    for i in range(test.shape[0]):
        case = ""
        case = case + "0 "
        case = case + features[test.ix[i, "item_id"]]
        case = case + "3:" + str(test.ix[i, "user_id"]) + ":1 "
        case = case + "4:" + str(test.ix[i, "item_id"]) + ":1 "
        case = case + "5:" + str(test.ix[i, "day"]) + ":1 "
        case = case + "6:" + str(test.ix[i, "time"]) + ":1 "
        case = case + "\n"
        out2.writelines(case)
        if(i % 1000 == 0):
            print(i)
    
    out2.close()
    '''
    for i in range(train.shape[0]):
        case = ""
        case = case + str(train.ix[i, 'score'])
        case = case + features[train.ix[i, "item_id"]]
        case = case + "3:" + str(train.ix[i, "user_id"]) + ":1 "
        case = case + "4:" + str(train.ix[i, "item_id"]) + ":1 "
        case = case + "5:" + str(train.ix[i, "day"]) + ":1 "
        case = case + "6:" + str(train.ix[i, "time"]) + ":1 "
        case = case + "\n"
        out1.writelines(case)
        if(i % 1000 == 0):
            print(i)
'''   
    
Construct_Mat()
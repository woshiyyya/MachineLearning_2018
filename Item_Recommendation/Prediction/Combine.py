import pandas

def Write_Ans(pred):
    sample = pandas.read_csv("3171.csv")
    sample['pred'] = pred
    sample.to_csv("Submission.csv",index = False)
    return

def Combine():
    file = ["best.csv","submit.csv"]
    weight = [0.5, 0.5]
    pred = pandas.read_csv("3171.csv")['pred'] * 0
    for i in range(len(file)):
        pred = pred + pandas.read_csv(file[i])['pred'] * weight[i]
    #pred /= len(file)
    Write_Ans(pred)
    return

Combine()

import pandas

def Write_Ans(pred):
    sample = pandas.read_csv("sample_submission.csv")
    ID = sample['id']
    DF = pandas.DataFrame({'id':ID,'pred':pred})
    DF.to_csv("Submission.csv",index = False)
    return

def Combine():
    file = ["918.csv","9216.csv","New_Staking.csv"]
    pred = pandas.read_csv("sample_submission.csv")['pred'] * 0
    for f in file:
        pred = pred + pandas.read_csv(f)['pred']
    pred /= len(file)
    Write_Ans(pred)
    return

Combine()

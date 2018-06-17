import pandas

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

Write_Ans("pred.txt", "submission.csv")
#coding=utf-8
import json
import time
import re
import jieba
import pickle
from bs4 import BeautifulSoup
import pandas


file = open('train.json','r')
Write = open('write.txt','w',encoding = 'utf-8')
trainy = pandas.read_csv("sample_submission.csv")['pred']
i = 2
cnt = 0
for line in file:
    if trainy[cnt] == 1:
        Dict = json.loads(line)
        Soup = BeautifulSoup(Dict['content'],'lxml')
        case = Soup.get_text()
        #case = case.encode("utf-8").decode("unicode_escape")
        Write.write(str(i)+"       ")
        Write.write(case)
        Write.write("\n\n")
        if i > 500:
            break;
        i = i + 1
    cnt = cnt + 1
    
Write.close()
file.close()

import json
import time
import re
import jieba
import pickle
from bs4 import BeautifulSoup
'''
train_content_file = open("train_content.pkl",'wb')
train_id_file = open("train_id.pkl",'wb')
train_title_file = open("train_title.pkl",'wb')
'''

train_content_file = open("test_content_nodup.pkl",'wb')
train_id_file = open("test_id.pkl",'wb')
train_title_file = open("test_title.pkl",'wb')

sw_file = open("StopWords.txt",'r',encoding = 'UTF-8')
full_word = []
full_id = []
full_title = []

stop_words = sw_file.read().splitlines()
for i in range(len(stop_words)):
    stop_words[i] = stop_words[i].strip()

#with open("train.json", 'r') as f:
with open("test.json", 'r') as f:
    i= 0
    for line in f:
        i = i + 1
        if i%1000 == 0:
            print("Now processed %d:",i)
            
        Dict = json.loads(line)
        full_id.append(Dict['id'])
        full_title.append(Dict['title'])
        
        soup = BeautifulSoup(Dict['content'],'lxml')
        content = soup.get_text()
        case = list(jieba.cut(content))
        #case = list(set(case))
        for sw in stop_words:
            if sw in case:
                case.remove(sw)
        pickle.dump(case, train_content_file)

#pickle.dump(full_id, train_id_file)
#pickle.dump(full_title, train_title_file)

train_id_file.close()
train_title_file.close()
train_content_file.close()

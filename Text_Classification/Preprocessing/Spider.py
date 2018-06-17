# coding=utf-8
from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import pickle
import chardet

site = "https://www.baidu.com/s?ie=UTF-8&wd="
train_file = open('train_title.pkl','rb')
train_num = 321910
test_num = 40239
title = pickle.load(train_file)
for i in range(train_num):
    t = title[i].encode('unicode-escape')
    HTML = urlopen(site+str(t))
    soup = BeautifulSoup(HTML,'lxml')
    print(soup.get_text())
    Ans_site = soup.findAll('h3',{"class":'t'})
    SiteList = []
    for x in Ans_site:
        n = x.a['href']
        print(n)
        SiteList.append(n)


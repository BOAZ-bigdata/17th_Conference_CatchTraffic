import glob
import pandas as pd
import os
from tqdm import tqdm
import re

area_list = os.listdir('./역정리')


# # txt 파일 합치기

df = pd.DataFrame()
areas = []
stations = []
error = []

for area in tqdm(area_list):
    station_list = os.listdir('./역정리/{}'.format(area))
    for station in station_list :
        path_list = glob.glob(r'C:\Users\82109\project\boaz\역정리\{}\{}\*.txt'.format(area,station))
        li = []
        for i in range(len(path_list)):
            t = path_list[i]
            f = open(t, 'r', encoding = 'utf-8')
            line = f.readline()
            li.append(line)
            f.close()
        try:
            li = li[:900]
            df[station[1:]] = li
            areas.append(area)
            stations.append(station)
        except:
            error.append('{}-{}-{}개'.format(area, station,len(li)))

df.to_csv(r'C:\Users\82109\project\boaz\crawling.csv', encoding = 'utf-8-sig', index = False)


# # 해시태그 split하기
df = pd.read_csv('crawling.csv')

data = pd.DataFrame()
for n in tqdm(stations) :
    lists = [re.sub('[^A-Za-z0-9가-힣#]', '', s) for s in df[n[1:]]]  # 숫자/문자/#만 남기고 없애서 리스트 하나로 만들기
    lists = [s for s in lists if "#" in s]  # 해시태그가 있는 게시물만 가져오기
    lists = '>>>'.join(lists) # 게시물끼리 구분 주기 위해 >>>으로 합치기
    lists = lists.replace('#','___#') # split('#')으로 자르면 해시태그 사라지니까 # 표시 남기기 위해 구분자 추가
    lists = lists.split('___')
    lists_t = [s for s in lists if ">>>" not in s] # '>>>' 없는 요소 따로 뺴내기
    lists = [s.split('>>>') for s in lists if ">>>" in s] # 이전게시물이 해시태그, 이후게시물이 글이면 >>>로 붙어있을 것! 이거 자르기
    lists = [s[0] for s in lists] # 해시태그 값만 가져오기
    lists = lists + lists_t # lists와 lists_t 합치기
    lists = [' '.join(lists)]
    data[n[1:]] = lists

data = data.T
data['area'] = areas

data.to_csv(r'C:\Users\82109\project\boaz\hashtag.csv', encoding = 'utf-8-sig')

# # parquet 변환

dataframe = pd.read_csv(r'C:\Users\82109\project\boaz\hashtag.csv', encoding = 'utf-8-sig')
dataframe

dataframe.columns = ['station','hashtag', 'area']

# get_ipython().system('pip install fastparquet')

dataframe.to_parquet('hashtag.parquet')

pd.read_parquet('hashtag.parquet')




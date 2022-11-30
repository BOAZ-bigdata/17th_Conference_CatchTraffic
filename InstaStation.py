# pandas, tqdm, fastparquet, python-snappy 설치하기
import glob
import pandas as pd
import os
from tqdm import tqdm
import re

# 해시태그 split
def split_hashtag(areas, stations, df, save_dir):
    data = pd.DataFrame()
    for n in tqdm(stations):
        lists = [re.sub('[^A-Za-z0-9가-힣#]', '', s) for s in df[n[1:]]]  # 숫자/문자/#만 남기고 없애서 리스트 하나로 만들기
        lists = [s for s in lists if "#" in s]  # 해시태그가 있는 게시물만 가져오기
        lists = '>>>'.join(lists)  # 게시물끼리 구분 주기 위해 >>>으로 합치기
        lists = lists.replace('#', '___#')  # split('#')으로 자르면 해시태그 사라지니까 # 표시 남기기 위해 구분자 추가
        lists = lists.split('___')
        lists_t = [s for s in lists if ">>>" not in s]  # '>>>' 없는 요소 따로 뺴내기
        lists = [s.split('>>>') for s in lists if ">>>" in s]  # 이전게시물이 해시태그, 이후게시물이 글이면 >>>로 붙어있을 것! 이거 자르기
        lists = [s[0] for s in lists]  # 해시태그 값만 가져오기
        lists = lists + lists_t  # lists와 lists_t 합치기
        lists = [' '.join(lists)]
        data[n[1:]] = lists
    data = data.T
    data['area'] = areas
    data = data.reset_index()
    data.columns = ['station', 'hashtag', 'area']
    data.to_csv(save_dir + '/hashtag.csv', encoding='utf-8-sig')
    data.to_parquet(save_dir + '/hashtag.parquet')

# txt 파일 합치고 split 전처리
def sum_txt(area_dir, save_dir):
    area_list = os.listdir(area_dir)
    df = pd.DataFrame()
    areas = []
    stations = []
    error = []

    for area in tqdm(area_list):
        station_list = os.listdir(area_dir + '/' + area)
        for station in station_list :
            path_list = glob.glob(area_dir + '/' + area + '/' + station + '/*.txt')
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
    print('오류 폴더의 데이터 개수 : ')
    print(error)
    df.to_csv(save_dir + '/crawling_df.csv', encoding = 'utf-8-sig', index = False)
    split_hashtag(areas, stations, df, save_dir)

print('경로 입력 예시 : C:/Users/82109/project/boaz/역정리')
area_dir = input('역정리 폴더 절대경로 입력 : ')
save_dir = input('결과 데이터 저장 폴더 절대경로 입력: ')
sum_txt(area_dir, save_dir)

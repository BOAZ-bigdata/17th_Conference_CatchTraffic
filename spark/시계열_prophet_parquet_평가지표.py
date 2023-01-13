#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google import colab
colab.drive.mount("/content/drive")


# In[ ]:


import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# In[ ]:


data = "/content/drive/MyDrive/Colab Notebooks/엔지어드브/"
df=pd.read_csv(f'{data}서울교통공사_역별 일별 시간대별 승하차인원 정보_20220831.csv', encoding='cp949')


# In[ ]:


import timeit
import datetime


# In[ ]:





# In[ ]:


df.drop(['24시이후'], axis=1, inplace=True)


# In[ ]:


# 일단 호선이 하나인 역 골라서 확인해보기
df = df.loc[df.승하차구분=='승차']


# In[ ]:


df.reset_index(drop=True, inplace=True)


# In[ ]:


df1 = df.copy()
df2 = df.copy()
df3 = df.copy()
df4 = df.copy()
df5 = df.copy()
df6 = df.copy()
df7 = df.copy()
df8 = df.copy()
df9 = df.copy()
df10 = df.copy()
df11 = df.copy()
df12 = df.copy()
df13 = df.copy()
df14 = df.copy()
df15 = df.copy()
df16 = df.copy()
df17 = df.copy()
df18 = df.copy()
df19 = df.copy()


# In[ ]:


df1 = df1[['수송일자','06시이전','호선','역명']]
df2 = df2[['수송일자','06-07시간대','호선','역명']]
df3 = df3[['수송일자','07-08시간대','호선','역명']]
df4 = df4[['수송일자','08-09시간대','호선','역명']]
df5 = df5[['수송일자','09-10시간대','호선','역명']]
df6 = df6[['수송일자','10-11시간대','호선','역명']]
df7 = df7[['수송일자','11-12시간대','호선','역명']]
df8 = df8[['수송일자','12-13시간대','호선','역명']]
df9 = df9[['수송일자','13-14시간대','호선','역명']]
df10 = df10[['수송일자','14-15시간대','호선','역명']]
df11 = df11[['수송일자','15-16시간대','호선','역명']]
df12 = df12[['수송일자','16-17시간대','호선','역명']]
df13 = df13[['수송일자','17-18시간대','호선','역명']]
df14 = df14[['수송일자','18-19시간대','호선','역명']]
df15 = df15[['수송일자','19-20시간대','호선','역명']]
df16 = df16[['수송일자','20-21시간대','호선','역명']]
df17 = df17[['수송일자','21-22시간대','호선','역명']]
df18 = df18[['수송일자','22-23시간대','호선','역명']]
df19 = df19[['수송일자','23-24시간대','호선','역명']]


# In[ ]:


df1.columns = ['수송일자', '인원','호선','역명']
df2.columns = ['수송일자', '인원','호선','역명']
df3.columns = ['수송일자', '인원','호선','역명']
df4.columns = ['수송일자', '인원','호선','역명']
df5.columns = ['수송일자', '인원','호선','역명']
df6.columns = ['수송일자', '인원','호선','역명']
df7.columns = ['수송일자', '인원','호선','역명']
df8.columns = ['수송일자', '인원','호선','역명']
df9.columns = ['수송일자', '인원','호선','역명']
df10.columns = ['수송일자', '인원','호선','역명']
df11.columns = ['수송일자', '인원','호선','역명']
df12.columns = ['수송일자', '인원','호선','역명']
df13.columns = ['수송일자', '인원','호선','역명']
df14.columns = ['수송일자', '인원','호선','역명']
df15.columns = ['수송일자', '인원','호선','역명']
df16.columns = ['수송일자', '인원','호선','역명']
df17.columns = ['수송일자', '인원','호선','역명']
df18.columns = ['수송일자', '인원','호선','역명']
df19.columns = ['수송일자', '인원','호선','역명']


# In[ ]:


df1['수송일자'] = df1['수송일자'] + ' 05'
df2['수송일자'] = df2['수송일자'] + ' 06'
df3['수송일자'] = df3['수송일자'] + ' 07'
df4['수송일자'] = df4['수송일자'] + ' 08'
df5['수송일자'] = df5['수송일자'] + ' 09'
df6['수송일자'] = df6['수송일자'] + ' 10'
df7['수송일자'] = df7['수송일자'] + ' 11'
df8['수송일자'] = df8['수송일자'] + ' 12'
df9['수송일자'] = df9['수송일자'] + ' 13'
df10['수송일자'] = df10['수송일자'] + ' 14'
df11['수송일자'] = df11['수송일자'] + ' 15'
df12['수송일자'] = df12['수송일자'] + ' 16'
df13['수송일자'] = df13['수송일자'] + ' 17'
df14['수송일자'] = df14['수송일자'] + ' 18'
df15['수송일자'] = df15['수송일자'] + ' 19'
df16['수송일자'] = df16['수송일자'] + ' 20'
df17['수송일자'] = df17['수송일자'] + ' 21'
df18['수송일자'] = df18['수송일자'] + ' 22'
df19['수송일자'] = df19['수송일자'] + ' 23'

df1['수송일자'] = df1['수송일자'].astype('datetime64')
df2['수송일자'] = df2['수송일자'].astype('datetime64')
df3['수송일자'] = df3['수송일자'].astype('datetime64')
df4['수송일자'] = df4['수송일자'].astype('datetime64')
df5['수송일자'] = df5['수송일자'].astype('datetime64')
df6['수송일자'] = df6['수송일자'].astype('datetime64')
df7['수송일자'] = df7['수송일자'].astype('datetime64')
df8['수송일자'] = df8['수송일자'].astype('datetime64')
df9['수송일자'] = df9['수송일자'].astype('datetime64')
df10['수송일자'] = df10['수송일자'].astype('datetime64')
df11['수송일자'] = df11['수송일자'].astype('datetime64')
df12['수송일자'] = df12['수송일자'].astype('datetime64')
df13['수송일자'] = df13['수송일자'].astype('datetime64')
df14['수송일자'] = df14['수송일자'].astype('datetime64')
df15['수송일자'] = df15['수송일자'].astype('datetime64')
df16['수송일자'] = df16['수송일자'].astype('datetime64')
df17['수송일자'] = df17['수송일자'].astype('datetime64')
df18['수송일자'] = df18['수송일자'].astype('datetime64')
df19['수송일자'] = df19['수송일자'].astype('datetime64')


# In[ ]:


result = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,
                    df11,df12,df13,df14,df15,df16,df17,df18,df19])


# In[ ]:


result.info()


# In[ ]:


result = result.sort_values('수송일자')


# In[ ]:


result = result.astype({'호선':'str'})


# In[ ]:


result['역명']=result['역명'] +' '+ result['호선']


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


label = LabelEncoder()
result['역명라벨'] = label.fit_transform(result.역명)


# In[ ]:


result


# In[ ]:


result = result.reset_index(drop=True)


# In[ ]:


df = result[['수송일자', '인원', '역명라벨']]


# In[ ]:


df.rename(columns={'수송일자':'ds', '인원':'y'}, inplace=True)


# In[ ]:


df.to_csv('ps.csv', index=False)


# In[ ]:


get_ipython().system('pip install pystan~=2.14')
get_ipython().system('pip install fbprophet')


# In[ ]:


get_ipython().system('pip install yfinance')


# In[ ]:


from fbprophet import Prophet 
from prophet.plot import add_changepoints_to_plot


# # prophet 1개

# In[ ]:





# In[ ]:


result1 = result[(result.역명 == '명동')&(result.수송일자>'2022-07-01')]


# In[ ]:


result1.reset_index(drop=True, inplace=True)


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


sns.set(rc={'figure.figsize':(25,10)})
sns.lineplot(x=result1['수송일자'] , y=result1['인원'])


# In[ ]:


temp_df = result1 # 날짜 기간 나누기 
study_df = pd.DataFrame() # 빈데이터 프레임 생성 
study_df['ds'] = temp_df.reset_index(drop=False)['수송일자'] # 시간 부분 넣기 
study_df['y'] =temp_df.reset_index(drop=False)['인원'] # 최대 전력 수요량 넣기 


# In[ ]:


study_df # 데이터셋 준비


# In[ ]:


model = Prophet() # 모델 인스턴스 선언 
model.fit(study_df) # 모델 생성 

future = model.make_future_dataframe(periods=1000, freq='H') #100개 예측 
forecast = model.predict(future) # 예측 결과 생성 
fig = model.plot(forecast) # 예측결과 확인 


# In[ ]:


fig = model.plot_components(forecast)


# # 스파크

# In[ ]:


get_ipython().system('pip install pyspark')


# In[ ]:


import pyspark


# In[ ]:


from pyspark.sql import SparkSession


# In[ ]:


from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import *
from pyspark.sql.functions import col


# In[ ]:


get_ipython().system('pip install fastparquet')


# In[ ]:


df = pd.read_csv('ps.csv')


# In[ ]:


df.y.max()


# In[ ]:


start = timeit.default_timer() 
df.to_parquet('my.parquet')
stop = timeit.default_timer()

print('time: ', stop - start)


# # pandasUDF

# In[ ]:


spark=SparkSession.builder.appName('Practise').getOrCreate()


# In[ ]:


start = timeit.default_timer() 
df_pyspark = spark.read.parquet('my.parquet',header=True, inferSchema=True)
stop = timeit.default_timer()

print('time: ', stop - start)


# In[ ]:





# In[ ]:


df_pyspark = df_pyspark.withColumn("ds", df_pyspark["ds"].cast("timestamp"))
df_pyspark = df_pyspark.withColumn("y", df_pyspark["y"].cast("integer"))
df_pyspark = df_pyspark.withColumn("역명라벨", df_pyspark["역명라벨"].cast("integer"))


# In[ ]:


df_pyspark.printSchema()


# In[ ]:


df_pyspark.show()


# In[ ]:


# 데이터 분할 
df_pyspark.createOrReplaceTempView("item_sales") 
sql = "select * from item_sales" 
sales_part = (spark.sql(sql)
   .repartition(spark.sparkContext.defaultParallelism, 
   ['역명라벨'])).cache( ) 
sales_part.explain()


# In[ ]:


# 데이터 분할 
df_pyspark.createOrReplaceTempView("item_sales") 
sql = "select * from item_sales" 
sales_part = (spark.sql(sql)
   .repartition(spark.sparkContext.defaultParallelism, 
   ['역명라벨']))   # 위에거에서 .cache()만 뺀거임 
sales_part.explain()


# In[ ]:


schema = StructType([
                     StructField('역명라벨', IntegerType()),
                     StructField('ds', TimestampType()),
                     StructField('y', IntegerType()),                                       
                     StructField('yhat', DoubleType()), 
                     StructField('yhat_upper', DoubleType()), 
                     StructField('yhat_lower', DoubleType()),
                     
])


# In[ ]:


@pandas_udf(schema, PandasUDFType.GROUPED_MAP)
def forecast_store_item(history_pd):
  model = Prophet() # 모델 인스턴스 선언 
  model.fit(history_pd) # 모델 생성 

  future = model.make_future_dataframe(periods=1000, freq='H') #100개 예측 
  forecast = model.predict(future) # 예측 결과 생성 


  f_pd = forecast[['ds', 'yhat','yhat_upper','yhat_lower']] 
  st_pd = history_pd[['ds', '역명라벨', 'y',]] 
  result_pd = f_pd.join(st_pd.set_index('ds'), on='ds', how='left')
  # 저장소 및 항목 채우기 
  result_pd['역명라벨'] = history_pd['역명라벨'].iloc[0]
  return result_pd[['역명라벨', 'ds', 'y', 'yhat',
                    'yhat_upper','yhat_lower']]


# In[ ]:


# 모든 역에 함수를 적용 
results = sales_part.groupby(['역명라벨']).apply(forecast_store_item)

start = timeit.default_timer() 
results.show() 
stop = timeit.default_timer()


# In[ ]:


print('Time: ', stop - start)


# In[ ]:


import fastparquet


# In[ ]:


results.filter(results['ds']>"2022-09-01").write.parquet("results.parquet")


# In[ ]:


results.filter(results['ds']>"2022-09-01").write.csv("results.csv")


# In[ ]:


a = pd.read_parquet('results.parquet')


# In[ ]:


a.to_csv('ss.csv', index=False)


# In[19]:


data = "/content/drive/MyDrive/Colab Notebooks/엔지어드브/"
result=pd.read_csv(f'{data}결과.csv')
label=pd.read_csv(f'{data}역명라벨.csv')


# In[20]:


label.호선.value_counts()


# In[ ]:


label.head()


# In[ ]:


label = label[['역명', '역명라벨']]


# In[ ]:


label.groupby(['역명', '역명라벨'], as_index=False).mean()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


results.write.parquet("results.parquet")


# In[ ]:


a = pd.read_parquet('results.parquet')


# In[ ]:


a.to_csv('프로펫_결과.csv', index = False)


# In[ ]:


a


# In[ ]:


def xx(x):
  if x < 0:
    return 0
  else:
    return x

a['yy'] = a.yhat.apply(xx)


# In[ ]:


aa =  a[a.y.isnull()==False]


# In[ ]:





# In[ ]:


from sklearn.metrics import mean_absolute_error
mean_absolute_error(aa.y, aa.yy)


# In[ ]:


from sklearn.metrics import mean_squared_error 
mean_squared_error(aa.y, aa.yy)


# In[ ]:


import numpy as np
MSE = mean_squared_error(aa.y, aa.yy) 
np.sqrt(MSE)


# In[ ]:


from sklearn.metrics import mean_squared_log_error
mean_squared_log_error(aa.y, aa.yy)


# In[ ]:





# In[ ]:


https://kgw7401.tistory.com/m/74

https://data-newbie.tistory.com/m/891


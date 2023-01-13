#!/usr/bin/env python
# coding: utf-8

# In[13]:


# openjdk8 설치
get_ipython().system('apt-get install openjdk-8-jdk-headless -qq > /dev/null')
# spark3.2.1( hadoop3.2 ) tar 다운로드
get_ipython().system('wget -q https://archive.apache.org/dist/spark/spark-3.2.1/spark-3.2.1-bin-hadoop3.2.tgz')
# spark3.2.1( hadoop3.2 ) tar 다운로드 압축풀기
get_ipython().system('tar -xvf spark-3.2.1-bin-hadoop3.2.tgz')
# findspark 설치
get_ipython().system('pip install -q findspark')


# In[14]:


# 환경변수 설정
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-3.2.1-bin-hadoop3.2"


# In[15]:


# findspark를 통해 sparksession 정보를 찾아낼 수 있음
import findspark
findspark.init()
from pyspark.sql import SparkSession


# 데이터프레임 사용을 위한 sparksession 생성
spark = SparkSession.builder.master("local[*]").getOrCreate()


# 

# In[22]:


# 파일 업로드
from google.colab import files
myfile = files.upload()


# In[25]:


# 테스트를 위한 csv 파일 불러오기
hashtag_df=spark.read.option('header','true').csv('hashtag (3).csv',inferSchema=True)


# In[26]:


hashtag_df.show()


# In[33]:


from pyspark.sql.functions import monotonically_increasing_id

hashtag_df = hashtag_df.withColumn("index", monotonically_increasing_id())

hashtag_df.show()


# 

# In[73]:


# 데이터프레임 확인
import pyspark.sql.functions as f
from pyspark.sql.functions import lit

columns = ['stations', 'hashtags', 'area']
result_df = hashtag_df.filter((hashtag_df['index']==0))
result_df = result_df.withColumn('word', f.explode(f.split(f.col('hashtags'), '#'))).groupBy('word').count().sort('count', ascending=False)
result_df = result_df.withColumn("station", lit(hashtag_df.filter(hashtag_df['index']==0).collect()[0].__getitem__('station')))
for i in range(hashtag_df.count()):
  tmp_df = hashtag_df.filter((hashtag_df['index']==i))
  tmp_df = tmp_df.withColumn('word', f.explode(f.split(f.col('hashtags'), '#'))).groupBy('word').count().sort('count', ascending=False)
  tmp_df = tmp_df.withColumn("station", lit(hashtag_df.filter(hashtag_df['index']==i).collect()[0].__getitem__('station')))
  # tmp_df.show()
  result_df = result_df.union(tmp_df)
result_df.show()


# 
# 
# ---
# 
# 

# In[79]:


result_df.coalesce(1).write.csv("results2.csv")


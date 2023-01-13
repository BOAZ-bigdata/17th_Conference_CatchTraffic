#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#깃헙: https://github.com/instaloader/instaloader
#공식 사이트: https://instaloader.github.io/
#참고 블로그: https://martechwithme.com/download-instagram-posts-stories-hashtags-highlights-python/


# In[1]:


# 버전은 4.9.5 이상으로 설치해야 오류 안남

#pip install instaloader==4.9.5


# In[ ]:


# 로컬에서 돌려야 돌아감


# In[1]:


import instaloader
from itertools import dropwhile, takewhile
from datetime import datetime
from instaloader import *


# In[2]:





# In[2]:


# 다운로드할 때 사진이나 영상은 제외
instance = instaloader.Instaloader(download_pictures=False,download_videos=False)

# 인스타 로그인(일부 기능에 필요)
#instance.login(user="",passwd="")


# In[3]:


# 맛집 해시태그 n개 다운로드(최신 기준)
instance.download_hashtag(hashtag="홍대입구",max_count=1000)


# In[7]:


# 해시태그 인기 게시물 가져오기

hashtag = instaloader.Hashtag.from_name(instance.context, "맛집")
posts = hashtag.get_top_posts()

for post in posts:
    instance.download_post(post, target='귀여운폴더')


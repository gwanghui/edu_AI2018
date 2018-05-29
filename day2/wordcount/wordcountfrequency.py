import pandas as pd # 필요 라이브러리 불러오기
from collections import Counter
import pytagcloud

# 데이터 불러오기
news = pd.read_csv("./ai_news_keyword_2013.CSV", engine='python')

news = news['키워드'] # 키워드 추출
news = news.values.tolist()
keywords = []

for n in news:
    keywords = keywords + n.split(',')
    
count = Counter(keywords) # 각 단어 별 개수 계산하기
tags = count.most_common(200) # 상위 200개 추출

taglist = pytagcloud.make_tags(tags, maxsize=60) # wordcloud 만들기
pytagcloud.create_tag_image(taglist, 'wordcloud.jpg', fontname='Malgun Gothic', rectangular=False)

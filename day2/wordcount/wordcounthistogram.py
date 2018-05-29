import pandas as pd # 필요 라이브러리 불러오기
import plotly
import plotly.graph_objs as go
from collections import Counter

news = pd.read_csv("./ai_news_keyword_2013.CSV", engine='python') # 데이터 불러오기

news = news['키워드'] # 키워드 추출
news = news.values.tolist()
keywords = []
for n in news:
    keywords = keywords + n.split(',')
    
keyword_counts = Counter(keywords) # 각 단어 별 개수 계산하기
keywords_top100 = keyword_counts.most_common(100) # 상위 100개 추출

word = []
counts = []

for key in keywords_top100:
    word.append(key[0])
    counts.append(key[1])
    
data = [go.Bar(x=word, # histogram 만들기
                y=counts)]

filename = 'doc-histogram'
plotly.offline.plot(data, filename=filename + '.html') # 만들어진 histogram 저장

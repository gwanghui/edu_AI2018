import pandas as pd # 필요 라이브러리 불러오기
import plotly
import plotly.graph_objs as go

iris = pd.read_csv('Iris.csv') # Iris 데이터셋 불러오기
print(iris.head()) # 데이터 확인

features = list(iris) # Iris의 특성 이름 저장
features.remove('Species')
print(features) # 특성 확인

plots = [] # 특성 별 boxplot 만들기
for feature in features:
  plot = go.Box(
    y=iris[feature],
    name = feature,
  )
  plots.append(plot)
  
# 만들어진 boxplot 저장
plotly.offline.plot(plots, filename='Iris-boxplot.html')

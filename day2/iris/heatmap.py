import pandas as pd # 필요 라이브러리 불러오기
import plotly
import plotly.graph_objs as go

iris = pd.read_csv('Iris.csv') # Iris 데이터셋 불러오기

# Iris의 특성 이름 저장
features = list(iris.corr())

# 특성들의 상관관계 테이블 저장
table = iris.corr().values.tolist()

# heatmap 만들기
heatmap = go.Heatmap(z=table,
                    x=features,
                    y=features)
data=[heatmap]

# 만들어진 heatmap 저장
filename = 'Iris-heatmap'
plotly.offline.plot(data, filename=filename + '.html')

import pandas as pd # 필요 라이브러리 불러오기
import plotly
import plotly.graph_objs as go

accidents = pd.read_csv('Traffic_Accident_20161019.csv', engine='python', header=None, index_col=0) # Iris 데이터셋 불러오기
accidents_t = accidents.T;

features = list(accidents_t)
features.remove('분류')

print(features)

for feature in features:
    print(accidents_t[feature])
    plot = go.Box(
        y=accidents_t[feature],
        name = feature,
    )
    plots.append(plot)

  
# 만들어진 boxplot 저장
plotly.offline.plot(plots, filename='accident-boxplot.html')

import pandas as pd # 필요 라이브러리 불러오기
import plotly
import plotly.graph_objs as go
iris = pd.read_csv('Iris.csv') # Iris 데이터셋 불러오기
features = list(iris) # Iris의 특성 이름 저장
features.remove('Species')
all_species = ['setosa', 'versicolor', 'virginica'] # Iris의 종류 이름 저장
for feature in features: # histogram 만들기
    data = []
    for species in all_species:
        feature_data = iris[feature]
        trace = go.Histogram(
            x=feature_data[iris.Species == species],
            opacity=0.75, # 투명도
            name=species)
        data.append(trace)
    layout = go.Layout(
        barmode='overlay', # 겹쳐서 그리기
        xaxis=dict(title=feature+'(CM)'),
        yaxis=dict(title='Count'))
    fig = go.Figure(data = data, layout=layout)
    
    filename = 'Iris-histogram-'+feature
    plotly.offline.plot(fig, filename=filename+'.html') # 만들어진 histogram 저장

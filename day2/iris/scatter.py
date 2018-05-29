import pandas as pd # 필요 라이브러리 불러오기
import plotly
import plotly.graph_objs as go

iris = pd.read_csv('Iris.csv') # Iris 데이터셋 불러오기
features = list(iris) # Iris의 특성 이름 저장
features.remove('Species')
all_species = ['setosa', 'versicolor', 'virginica'] # Iris의 종류 이름 저장

for feature1 in features: # scatter 만들기
    for feature2 in features:
        if(feature1 == feature2):
            continue
        data = []
    for specie in all_species:
        feature1_data = iris[feature1]
        feature2_data = iris[feature2]
        trace = go.Scatter(
            x=feature1_data[iris.Species == specie],
            y=feature2_data[iris.Species == specie],
            mode='markers', marker=dict(size=14), name=specie )
        data.append(trace)
    layout = go.Layout(barmode='overlay', # 겹쳐서 그리기
        xaxis=dict(title=feature1),
        yaxis=dict(title=feature2))
    fig = go.Figure(data = data, layout=layout)
    filename = 'Iris-scatter-'+feature1+'-'+feature2 # 만들어진 scatter 저장
    plotly.offline.plot(fig, filename=filename+'.html', auto_open=False)

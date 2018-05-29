import pandas as pd # 필요 라이브러리 불러오기
import plotly
import plotly.graph_objs as go

iris = pd.read_csv('Iris.csv') # Iris 데이터셋 불러오기
features = list(iris) # Iris의 컬럼 이름 저장
label = [] # Iris종류를 문자에서 숫자로 바꾸어 저장
for feature in iris[features[4]]:
    if(feature == 'setosa'):
        label.append(1)
    elif (feature == 'versicolor'):
        label.append(2)
    elif (feature == 'virginica'):
        label.append(3)
        
pc = go.Parcoords( # PC 만들기
        line = dict(color = 'blue'),
        dimensions = list([
            dict(range = [1,3],
                tickvals = [1,2,3],
                label = features[4], values = label,
                ticktext = ['setosa', 'versicolor', 'virginica']),
            dict(label = features[0], values = iris[features[0]]),
            dict(label = features[1], values = iris[features[1]]),
            dict(label = features[2], values = iris[features[2]]),
            dict(label = features[3], values = iris[features[3]])]))

plotly.offline.plot([pc], filename='Iris-PC.html')

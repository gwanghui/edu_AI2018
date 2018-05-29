import pandas as pd # 필요 라이브러리 불러오기
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name) # 한글 폰트 설정

news = pd.read_csv("./ai_news_keyword_2013.CSV", engine='python') # 데이터 불러오기
news = news['키워드'] # 키워드 추출
news = news.values.tolist()

tfidf = TfidfVectorizer(max_features=100, max_df=0.95, min_df=0) # tf-idf 기준으로 뉴스 별
doc_tfidf = tfidf.fit_transform(news) # 상위 100개 단어 추출
tfidf_dict = tfidf.get_feature_names()

tsne = TSNE(n_components=2, n_iter=10000, verbose=1)
data_array = doc_tfidf.toarray()
Z = tsne.fit_transform(data_array.T) # 단어 기준으로 t-SNE 적용

plt.rcParams["figure.figsize"] = (10,10) # size 크게 키움

# scatter plot 만듦
plt.scatter(Z[:,0], Z[:,1])
for i in range(len(tfidf_dict)):
    plt.annotate(s=tfidf_dict[i], xy=(Z[i,0], Z[i,1]))

plt.show()

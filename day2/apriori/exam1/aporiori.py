import pandas as pd # 필요 라이브러리 불러오기
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import apriori

dataset = []
with open('./mart.csv', 'r') as reader:
    for line in reader:
        dataset.append(line.strip().split(','))
te = TransactionEncoder() # 거래 정보를 bag of word형식으로 저장
te_ary = te.fit(dataset).transform(dataset)

# 거래 정보를 pandas 데이터프레임으로 저장
df = pd.DataFrame(te_ary, columns=te.columns_)

# apriori 알고리즘을 이용하여 의미있는 규칙 찾기
frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.01)
print(rules) # 규칙 보기

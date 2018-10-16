UCI数据库是加州大学欧文分校(University of CaliforniaIrvine)提出的用于机器学习的数据库，这个数据库目前共有335个数据集，其数目还在不断增加，UCI数据集是一个常用的标准测试数据集。

```python
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer


mql = TfidfVectorizer(max_features=256).fit_transform(dfqa['question'].values)
```
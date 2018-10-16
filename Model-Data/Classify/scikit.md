# 1 决策树实现基本的分类
分类在大量语境下都非常重要，例如我们打算自动化一些决策过程，我们可以利用分类。在我们需要研究诈骗的情况下，有大量的事务，人去检查它们是不实际的，所以我们都可以使用分类自动化这种决策

```python
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier

X, y = datasets.make_classification(n_samples=1000, n_features=3, n_redundant=0)
dt = DecisionTreeClassifier()
dt.fit(X, y)
preds = dt.predict(X)
(y == preds).mean()
```
如果你观察dt对象，它拥有多种关键字参数，决定了对象的行为，我们如何选择对象十分重要，所以我们要详细观察对象的效果
我们要观察的第一个细节是max_depth。这是个重要的参数，决定了允许多少分支。决策树需要很长时间来生成样本外的数据，
它们带有一些类型的正则化会后，如何使用多种浅层决策树，来生成更好的模型。我们创建更复杂的数据集并观察当我们允许不同
max_depth时会发生什么。
```
import numpy as np

n_features = 200
X, y = datasets.make_classification(750, n_features, n_informative=5)
training = np.random.choice([True, False], p=[.75, .25], size=len(y))
accuracies = []
for x in np.arange(1, n_features+1):
    dt = DecisionTreeClassifier(max_depth=x)
    dt.fit(X[training], y[training])
    preds = dt.predict(X[~training])
    accuracies.append((preds == y[~training]).mean())
```
查看准确率曲线
```py
import matplotlib.pyplot as plt
%matplotlib inline

f, ax = plt.subplots(figsize=(7, 5))
ax.plot(range(1, n_features+1), accuracies, color='k')
ax.set_title('Decision Tree Accuracy')
ax.set_ylabel('% Correct')
ax.set_xlabel('Max Depth')
```
在较低的最大深度出得到了漂亮的准确率，让我们看看低级别的准确率
```python
N=15
f, ax = plt.subplots(figsize=(7, 5))
ax.plot(range(1, n_features+1)[:N], accuracies[:N], color='k')
ax.set_title('Decision Tree Accuracy')
ax.set_ylabel('% Correct')
ax.set_xlabel('Max Depth')
```
这个就是我们之前看到的峰值，比较令人惊讶的是它很快就下降了，

下面我们讨论熵和基尼系数之间的差异
熵不仅仅是给定变量的熵值，如果我们知道元素的值，它表示了熵中的变化，这叫做信息增益（IG），
```
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from io import StringIO
from sklearn import tree
import pydot
import pydotplus
from IPython.display import Image

X, y = datasets.make_classification(10000, 20, n_informative=3)
dt = DecisionTreeClassifier()
dt.fit(X, y)
# str_buffer = StringIO()
# tree.export_graphviz(dt, out_file=str_buffer)
# graph = pydotplus.graph_from_dot_data(str_buffer.getvalue())
# Image(graph.create_png()) 

dot_data = tree.export_graphviz(dt, out_file=None, 
#                          feature_names=iris.feature_names,  
#                          class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png()) 

```
你会看到有个非常深的树，八成是出现了过拟合，调整深度再来

```sh
brew install graphviz
pip install pyplotplus
```

```python
dt = DecisionTreeClassifier(max_depth=5)
dt.fit(X, y)
dot_data = tree.export_graphviz(dt, out_file=None, 
                     filled=True, rounded=True,  
                     special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png()) 
```

这棵树是不是简单多了,下面换一下分割标准
```python
dt = DecisionTreeClassifier(criterion='entropy', max_depth=5).fit(X, y)
dot_data = tree.export_graphviz(dt, out_file=None,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png()) 
```
很容易看到，前两个分割是相同的特征，之后的分割以相似总数分布，这是个良好的合理检查
第一个分割的熵是0.99，但是时候用基尼系数的时候是0.5

## 3. 使用许多决策树，随机森林
先来个实例
```python
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

X, y = datasets.make_classification(1000)
rf = RandomForestClassifier()
rf.fit(X, y)
print("Accuracy:\t {}".format((y==rf.predict(X)).mean()))
print("total Correct:\t {}".format((y==rf.predict(X)).sum()))
```
不仅仅是predict方法使用，我们也可以从独立的例子获取概率，这是个非常使用的特性，用于理解每个预测的不确定性。
```python
import pandas as pd
import matplotlib.pyplot as plt

probs = rf.predict_proba(X)
probs_df = pd.DataFrame(probs, columns=['0', '1'])
probs_df['was_correct'] = rf.predict(X) == y
f, ax = plt.subplots(figsize=(7, 5))
probs_df.groupby('0').was_correct.mean().plot(kind='bar', ax=ax)
ax.set_title('Accuracy at 0 class probality')
ax.set_ylabel('% Correct')
ax.set_xlabel('% tree for 0')
```

随机森林使用预定义数量的弱决策树，并且使用数据自己训练每一棵树，对于避免过拟合至关重要。
随机森林的另一个是特征重要性    
```python
f, ax = plt.subplots(figsize=(7, 5))
ax.bar(range(len(rf.feature_importances_)), rf.feature_importances_)
ax.set_title("Feature Importances")
```

## 4 调整随机森林模型
为了调整随机森林模型，我们首先需要创造数据集，它有一些难以预测，之后我们修改参数并且做一些预处理来更好地拟合数据集
```python
from sklearn import datasets
import numpy as np
from sklearn.ensemble import RandomForestClassifier

X, y = datasets.make_classification(n_samples=10000, 
                                    n_features=20, 
                                    n_informative=15,
                                    flip_y=.5, 
                                    weights=[.2, .8])
training = np.random.choice([True, False], p=[.8, .2], size=y.shape)
rf = RandomForestClassifier()
rf.fit(X[training], y[training])
preds = rf.predict(X[~training])
print("Accuracy:\t {}".format((preds == y[~training]).mean()))
```

我们要引入混淆军阵，作为模型评估度量之一
```python
from sklearn.metrics import confusion_matrix
import pandas as pd
import itertools
from matplotlib import pyplot as plt

max_feature_params = ['auto', 'sqrt', 'log2', 0.01, .5, .99]
confusion_matrixes = {}
for max_feature in max_feature_params:
    rf = RandomForestClassifier(max_features=max_feature)
    rf.fit(X[training], y[training])
    confusion_matrixes[max_feature] = confusion_matrix(y[~training], rf.predict(X[~training])).ravel()
```

```python
confusion_df = pd.DataFrame(confusion_matrixes)
f,ax = plt.subplots(figsize=(7, 5))
confusion_df.plot(kind='bar', ax=ax)
ax.legend(loc='best')
ax.set_title('Guessed vs Correct (i, j) where i is guess and j in the acture')
ax.grid()
# ax.set_xticklabels([str((i, j)) for i, j list(itertools.product(range(2), range(2)))])
ax.set_xlabel('Guessed vs Correct')
ax.set_ylabel('Correct')
```

## 5 使用支持向量机对数据进行分类
支持向量机是我们使用的技巧之一，原理是寻找一个平面，将数据集分割为组，并且是最优的。
```python
from sklearn import datasets
from sklearn.svm import SVC

X, y = datasets.make_classification()
base_svm = SVC()
base_svm.fit(X, y)
```
聚类通常和无监督技巧组合到一起， 这些技巧假设我们不知道结果变量，这会使结果模糊，以及实践客观。但是，聚类十分有用。我们会看到，我们可以使用聚类，将我们的估计在监督设置中本地化，这可能就是聚类非常高效的原因。
我们会浏览大量应用，从图像处理到回归以及离群点检测，通过这些应用，我们会看到聚类通常可以通过概率或者优化结构来观察，不同的解释会导致不同的权衡。我们会看到，如何训练模型，以便让工具尝试不同模型，在面对聚类问题的时候。

## 1 使用KMeans对数据聚类
聚类使一个非常实用的技巧，通常我们在采取行动时需要分治，考虑公司的潜在客户列表，公司可能需要将客户按类型分组，之后为这些分组划分职责，聚类可以使这个过程变得容易。KMeans可能是最知名的聚类算法之一，并且也是最知名的无监督学习技巧之一。
```python
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

blobs, classes = make_blobs(500, centers=3)
f, ax = plt.subplots(figsize=(7.5, 7.5))
rgb = np.array(['r', 'g', 'b'])
ax.scatter(blobs[:, 0], blobs[:, 1], color=rgb[classes])
ax.set_title("Blobs")
```

```python
from sklearn.cluster import KMeans
kmean = KMeans(n_clusters=3)
kmean.fit(blobs)
kmean.cluster_centers_
```

```python
f, ax = plt.subplots(figsize=(7.5, 7.5))
ax.scatter(blobs[:, 0], blobs[:, 1], color=rgb[classes])
ax.scatter(kmean.cluster_centers_[:, 0], kmean.cluster_centers_[:, 1],marker='*', s=250, color='black', label='Centers')
ax.set_title('Blobs')
ax.legend(loc='best')
```

### 工作原理
KMeans实际上是个非常简单的算法，它使簇中的点到均值的距离的平方和最小，首先它会设置一个预定义的簇数量K，之后执行这些事情
* 将每个数据点分配到最近的簇中
* 通过计算初中每个数据点的均值，更新每个形心
直到满足特定条件

### 2 优化形心数量
形心难以解释，并且也难以判断是否数量正确。理解你的数据是否是未分类的十分重要，因为这会直接影响到我们可用的评估手段。

#### 准备
为无监督学习评估模型表现是个挑战，所以在了解真实的情况的时候，sklearn拥有多种方式来评估聚类，但是不了解时就很少。
我们会以一个简单的簇模型开始，来评估它的相似度。这更多是出于机制的目的，因为测量一个簇的相似性在寻找簇数量的真实情况时显然没有用。
准备测试数据
```python
from sklearn.datasets import make_blobs
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics

blobs, classes = make_blobs(500, centers=3)
kmean = KMeans(n_clusters=3)
kmean.fit(blobs)
```

#### 查看轮廓（Silhouette）距离
轮廓距离是簇内不相似性，最近的簇间不相似性，以及这两个值最大值的比值，它可以看做簇间分离程度的度量。
让我们看一看数据点到形心的距离分布，理解轮廓距离非常有用。
```python
from sklearn import metrics
silhouette_samples = metrics.silhouette_samples(blobs, kmean.labels_)
np.column_stack((classes[:5],silhouette_samples[:5]))

f,ax = plt.subplots(figsize=(10, 5))
ax.set_title('Hist of Silhouette Samples')
ax.hist(silhouette_samples)
```

轮廓系数的均值通常用于描述整个模型的拟合度
```python
silhouette_samples.mean()
```
事实上metrics模块提供了一个函数来获得刚才的值，现在让我们训练多个簇的模型，看看平均分是什么样
```python
blobs, classes = make_blobs(500, centers=10)
sillhoutte_avgs = []
for k in range(2, 60):
    kmean = KMeans(n_clusters=k).fit(blobs)
    sillhoutte_avgs.append(metrics.silhouette_score(blobs, kmean.labels_))
    
f, ax = plt.subplots(figsize=(7, 5))
ax.plot(sillhoutte_avgs)
```
这个绘图表明，轮廓均值随着形心数量的变化情况，我们可以看到最优的数量是3，根据所生成的数据，但是最优的数量看起来是6或者7。这就是聚类的实际情况，十分普遍、我们不能获得正确的簇数量，我们只能估计数量的近似值

## 3 评估聚类的正确性
我们之前讨论了不知道真实情况的条件下的聚类评估，但是我们还没有讨论簇已知条件下的KMeans评估，在许多情况下，这都是不可知的，但是如果存在外部的标注，我们就会知道真实情况，或者至少是代理。


## 4 使用MiniBatch KMeans处理更多的数据
KMeans是一个不错的方法，但是不适合用于大量数据，是因为KMeans的复杂度，也就是说，我们可以使用更低的算法复杂度来获得近似解
MiniBatch Kmeans是KMeans的更快实现，KMeans的计算量非常大，问题是NPH的。
但是使用MiniBatch KMeans我们可以将KMeans加速几个数量级，者通过处理多个子样本来完成，它们叫做MiniBatch，如果子样本是收敛的，并且拥有良好的初始条件，就得到了常规KMeans的近似解。

MiniBatch聚类的性能概要分析
```python
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, MiniBatchKMeans

blobs, labels = make_blobs(int(1e6), 3)
kmeans = KMeans(n_clusters=3)
minibatch = MiniBatchKMeans(n_clusters=3)
```

运行速度比较
```python
%time kmeans.fit(blobs)
%time minibatch.fit(blobs)
```

两个形心的距离
```python
from sklearn.metrics import pairwise
pairwise.pairwise_distances(kmeans.cluster_centers_, minibatch.cluster_centers_)
```


## 5 使用KMeans聚类来量化图像
图像处理是个重要的话题，其中聚类有一些应用，值得指出的是Python中有一个非常不错的图像处理库。scikit-image是scikit-learn的姐妹项目

```python
from scipy import ndimage
img = ndimage.imread('./data/test.jpg')
plt.figure(figsize=(20, 8))
plt.imshow(img)
x, y, z = img.shape
long_img = img.reshape(x*y, z)
long_img.shape

from sklearn import cluster
k_means = cluster.KMeans(n_clusters=5)
k_means.fit(long_img)
```
未完待处理

## 6 寻找特征空间的最接近对象
有时候， 最简单的事情就是求出两个对象之间的距离，我们刚好需要寻找一些距离度量，计算成对距离(Pairwise)，并将结果与我们的预期比较。

## 8 将KMeans用于离群点检测
首先，我们会生成100个点的单个数据块，之后我们会识别5个离形心最远的点，它们就是潜在的离群点
```python
from sklearn.datasets import make_blobs
import numpy as np
from sklearn.cluster import KMeans

X, labels = make_blobs(100, centers=1)
kmeans = KMeans(n_clusters=1)
kmeans.fit(X)
```

画图显示
```python
f, ax = plt.subplots(figsize=(7, 5))
ax.set_title('Blob')
ax.scatter(X[:, 0], X[:, 1], label='Points')
ax.scatter(kmeans.cluster_centers_[:, 0],
          kmeans.cluster_centers_[:, 1], 
          label='Centroid', 
          color='r')
ax.legend()
```
现在，让我们识别5个最接近的点：
```python
distances = kmeans.transform(X)
sorted_idx = np.argsort(distances.ravel())[::-1][:5]
```

```python
f, ax = plt.subplots(figsize=(7, 5))
ax.set_label('Single Cluster')
ax.scatter(X[:, 0], X[:, 1], label='Points')
ax.scatter(kmeans.cluster_centers_[:,0],
           kmeans.cluster_centers_[:,1],
           label='Centroid', color='r')
ax.scatter(X[sorted_idx][:, 0], X[sorted_idx][:, 1], label='Extreme Value', edgecolors='g',
          facecolors='none', s=100)
ax.legend(loc='best')
```
删除离群点后，形心发生了变化
```python
new_X = np.delete(X, sorted_idx, axis=0)
new_kmeans = KMeans(n_clusters=1)
new_kmeans.fit(new_X)
f, ax = plt.subplots(figsize=(7, 5))
ax.set_title('Extreme Values Removed')
ax.scatter(new_X[:, 0], new_X[:, 1], label='Pruned Point')
ax.scatter(kmeans.cluster_centers_[:, 0],
          kmeans.cluster_centers_[:, 1],label='Old Centroid', color='r',s=80, alpha=0.5)
ax.scatter(new_kmeans.cluster_centers_[:, 0], 
          new_kmeans.cluster_centers_[:, 1], label='New Centroid', color='m', s=80, alpha=.5)
ax.legend(loc='best')
```
高斯分布和KMeans聚类之间有本质联系，让我们基于形心和样本的协方差军阵创建一个经验高斯分布，病查看每个点的概率，理论上是我们溢出的五个点，
```python
from scipy import stats
emp_dist = stats.multivariate_normal(kmeans.cluster_centers_.ravel())
lowest_prob_idx = np.argsort(emp_dist.pdf(X))[:5]
np.all(X[sorted_idx] == X[lowest_prob_idx])
```

## 9 将KNN用于回归
线性预测结果如下：
```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression

iris = load_iris()
iris.feature_names
X = iris.data
y = iris.target
lr = LinearRegression()
lr.fit(X, y)
print("The MSE is : {:.2}".format(np.power(y-lr.predict(X), 2).mean()))
```
现在，对于KNN回归，使用以下代码
```python

```
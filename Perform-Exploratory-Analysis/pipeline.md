本主题将再度释放管线命令的的光芒。之前我们用它处理缺失数据，只是牛刀小试，下面我们用管线命令把多个预处理步骤连接起来处理，会非常方便

```py
from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
iris_data = iris.data

mask = np.random.binomial(1, .25, iris_data.shape).astype(bool)
iris_data[mask] = np.nan
iris_data[:5]
```
本主题的目标是首先不全iris_data的缺失值，然后对补全的数据集用PCA，可以看出这个流程需要一个数据集合一个对照集合；管线命令会让事情变得简单
```py
from sklearn import pipeline, preprocessing, decomposition
pca = decomposition.PCA()
imputer = preprocessing.Imputer()
pipe = pipeline.Pipeline([('imputer', imputer), ('pca', pca)])
np.set_printoptions(2)
iris_data_transformed = pipe.fit_transform(iris_data)
iris_data_transformed[:5]
```
管线命令的每个步骤都是用一个元组表示，元组的第一个元素是对象的名称，第二个元素是对象

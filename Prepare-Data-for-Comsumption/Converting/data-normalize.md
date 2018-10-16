## 一.标准化的原因
    
通常情况下是为了消除量纲的影响。譬如一个百分制的变量与一个5分值的变量在一起怎么比较？只有通过数据标准化，都把它们标准到同一个标准时才具有可比性，一般标准化采用的是Z标准化，即均值为0，方差为1，当然也有其他标准化，比如0--1标准化等等，可根据自己的数据分布情况和模型来选择
## 二.适用情况
　　看模型是否具有伸缩不变性。不是所有的模型都一定需要标准化，有些模型对量纲不同的数据比较敏感，譬如SVM等。当各个维度进行不均匀伸缩后，最优解与原来不等价，这样的模型，除非原始数据的分布范围本来就不叫接近，否则必须进行标准化，以免模型参数被分布范围较大或较小的数据主导。但是如果模型在各个维度进行不均匀伸缩后，最优解与原来等价，例如logistic regression等，对于这样的模型，是否标准化理论上不会改变最优解。但是，由于实际求解往往使用迭代算法，如果目标函数的形状太“扁”，迭代算法可能收敛得很慢甚至不收敛。所以对于具有伸缩不变性的模型，最好也进行数据标准化。

## 三.三种数据变换方法的含义与应用
* Rescaling（重缩放/归一化）：通常是指增加或者减少一个常数，然后乘以/除以一个常数，来改变数据的衡量单位。例如：将温度的衡量单位从摄氏度转化为华氏温度。
* Normalizing（正则化）：通常是指除以向量的范数。例如：将一个向量的欧氏长度等价于1 。在神经网络中，“正则化”通常是指将向量的范围重缩放至最小化或者一定范围，使所有的元素都在[0,1]范围内。通常用于文本分类或者文本聚类中。
* Standardizing（标准化）：通常是为了消除不同属性或样方间的不齐性，使同一样方内的不同属性间或同一属性在不同样方内的方差减小。例如：如果一个向量包含高斯分布的随机值，你可能会通过除以标准偏差来减少均值，然后获得零均值单位方差的“标准正态”随机变量。

　　那么问题是，当我们在训练模型的时候，一定要对数据进行变换吗？这得视情况而定。很多人对多层感知机有个误解，认为输入的数据必须在[0,1]这个范围内。虽然标准化后在训练模型效果会更好，但实际上并没有这个要求。但是最好使输入数据中心集中在0周围，所以把数据缩放到[0,1]其实并不是一个好的选择。

　　如果你的输出激活函数的范围是[0,1](sigmoid函数的值域)，那你必须保证你的目标值也在这个范围内。但通常请款下，我们会使输出激活函数的范围适应目标函数的分布，而不是让你的数据来适应激活函数的范围。

　　当我们使用激活函数的范围为[0,1]时，有些人可能更喜欢把目标函数缩放到[0.1,0.9]这个范围。我怀疑这种小技巧的之所以流行起来是因为反向传播的标准化太慢了导致的。但用这种方法可能会使输出的后验概率值不对。如果你使用一个有效的训练算法的话，完全不需要用这种小技巧，也没有必要去避免溢出（overflow）

## 四.具体方法及代码
### 一）标准化
#### 1.1 scale----零均值单位方差
```py
from sklearn import preprocessing 
import numpy as np  
#raw_data
X = np.array([[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]])  
X_scaled = preprocessing.scale(X) 
#output
X_scaled = [[ 0.         -1.22474487  1.33630621]
             [ 1.22474487  0.         -0.26726124]
             [-1.22474487  1.22474487 -1.06904497]]
＃scaled之后的数据零均值，单位方差
X_scaled.mean(axis=0)  # column mean: array([ 0.,  0.,  0.])  
X_scaled.std(axis=0)  #column standard deviation: array([ 1.,  1.,  1.])
```

#### 1.2 StandardScaler----计算训练集的平均值和标准差，以便测试数据集使用相同的变换
```py
scaler = preprocessing.StandardScaler().fit(X) 
#out:
 StandardScaler(copy=True, with_mean=True, with_std=True)
scaler.mean_  
#out: 
array([ 1.,  0. ,  0.33333333])  
scaler.std_ 
#out:
 array([ 0.81649658,  0.81649658,  1.24721913]) 
#测试将该scaler用于输入数据，变换之后得到的结果同上
scaler.transform(X)
 #out: 
array([[ 0., -1.22474487,  1.33630621],  [ 1.22474487, 0. , -0.26726124],  [-1.22474487,1.22474487, -1.06904497]])  
scaler.transform([[-1., 1., 0.]])  
#scale the new data, out: 
array([[-2.44948974,  1.22474487, -0.26726124]])
```
注：
* 若设置with_mean=False 或者 with_std=False，则不做centering 或者scaling处理。
* scale和StandardScaler可以用于回归模型中的目标值处理。

### 二）归一化----将数据特征缩放至某一范围(scalingfeatures to a range)
另外一种标准化方法是将数据缩放至给定的最小值与最大值之间，通常是０与１之间，可用MinMaxScaler实现。或者将最大的绝对值缩放至单位大小，可用MaxAbsScaler实现。

使用这种标准化方法的原因是，有时数据集的标准差非常非常小，有时数据中有很多很多零（稀疏数据）需要保存住０元素。

#### 2.1 MinMaxScaler(最小最大值标准化)
　公式：X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0)) ; X_scaler = X_std/ (max - min) + min
```py
#例子：将数据缩放至[0, 1]间
X_train = np.array([[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]])
min_max_scaler = preprocessing.MinMaxScaler() 
X_train_minmax = min_max_scaler.fit_transform(X_train)  
#out: 
array([[ 0.5       ,  0.        ,  1.        ], 
[ 1.        ,  0.5       ,  0.33333333],        
[ 0.        ,  1.        ,  0.        ]])
#将上述得到的scale参数应用至测试数据
X_test = np.array([[ -3., -1., 4.]])  
X_test_minmax = min_max_scaler.transform(X_test) #out: array([[-1.5 ,  0. , 1.66666667]])
#可以用以下方法查看scaler的属性
min_max_scaler.scale_        #out: array([ 0.5 ,  0.5,  0.33...])
min_max_scaler.min_         #out: array([ 0.,  0.5,  0.33...])
```

#### 2.2 MaxAbsScaler（绝对值最大标准化）
与上述标准化方法相似，但是它通过除以最大值将训练集缩放至[-1,1]。这意味着数据已经以０为中心或者是含有非常非常多０的稀疏数据。
```py
X_train = np.array([[ 1., -1.,  2.],
                     [ 2.,  0.,  0.],
                    [ 0.,  1., -1.]])
max_abs_scaler = preprocessing.MaxAbsScaler()
X_train_maxabs = max_abs_scaler.fit_transform(X_train)
# out: 
array([[ 0.5, -1.,  1. ], [ 1. , 0. ,  0. ],       [ 0. ,  1. , -0.5]])
X_test = np.array([[ -3., -1.,  4.]])
X_test_maxabs = max_abs_scaler.transform(X_test) 
#out: 
array([[-1.5, -1. ,  2. ]])
max_abs_scaler.scale_  
#out: 
array([ 2.,  1.,  2.])
```
其实在scale模块里，也提供了这两种方法： minmax_scale和maxabs_scale

#### 2.3 对稀疏数据进行标准化

对稀疏数据进行中心化会破坏稀疏数据的结构，这样做没什么意义。但是我们可以对稀疏数据的输入进行标准化，尤其是特征在不同的标准时。MaxAbsScaler 和 maxabs_scale是专门为稀疏数据设计的，也是常用的方法。但是scale 和 StandardScaler只接受scipy.sparse的矩阵作为输入，并且必须设置with_centering=False。否则会出现 ValueError且破坏稀疏性，而且还会无意中分配更多的内存导致内存崩溃。RobustScaler不适用于稀疏数据的输入，但是你可以用 transform 方法。
　　scalers接受压缩的稀疏行（Compressed Sparse Rows）和压缩的稀疏列（Compressed Sparse Columns）的格式（具体参考scipy.sparse.csr_matrix 和scipy.sparse.csc_matrix）。其他的稀疏格式会被转化成压缩的稀疏行（Compressed Sparse Rows）格式。为了避免这种不必要的内存拷贝，推荐使用CSR或者CSC的格式。如果数据很小，可以在稀疏矩阵上运用toarray 方法。

#### 2.4 对离群点进行标准化

如果你的数据有离群点（上一篇我们提到过），对数据进行均差和方差的标准化效果并不好。这种情况你可以使用robust_scale 和 RobustScaler 作为替代。它们有对数据中心化和数据的缩放鲁棒性更强的参数。

## 三）正则化
### 3.1  L1、L2正则化

```py
x=np.array([[1.,-1.,2.],
            [2.,0.,0.],
            [0.,1.,-1.]])
x_normalized=preprocessing.normalize(x,norm='l2')
print(x_normalized)

# 可以使用processing.Normalizer()类实现对训练集和测试集的拟合和转换
normalizer=preprocessing.Normalizer().fit(x)
print(normalizer)
normalizer.transform(x)
```
注：稀疏数据输入：
normalize 和 Normalizer 既接受稠密数据（dense array-like），也接受稀疏矩阵（from scipy.sparse）作为输入
稀疏数据需要转换成压缩的稀疏行（Compressed Sparse Rows）格式（详见scipy.sparse.csr_matrix），为了避免不必要的内存拷贝，推荐使用CSR。

## 四）二值化
### 4.1特征二值化
特征二值化是把数值特征转化成布尔值的过程。这个方法对符合多变量伯努利分布的输入数据进行预测概率参数很有效。详细可以见这个例子sklearn.neural_network.BernoulliRBM.
此外，在文本处理中也经常会遇到二值特征值（很可能是为了简化概率推理），即使在实际中正则化后的词频或者TF-IDF的值通常只比未正则化的效果好一点点。
对于 Normalizer，Binarizer工具类通常是在Pipeline阶段（sklearn.pipeline.Pipeline）的前期过程会用到。下面举一个具体的例子：
```py
#input
X = [[ 1., -1.,  2.],
         [ 2.,  0.,  0.],
         [ 0.,  1., -1.]]
#binary
binarizer = preprocessing.Binarizer().fit(X)  # fit does nothing
binarizer
Binarizer(copy=True, threshold=0.0)
#transform
binarizer.transform(X)
#out:
array([[ 1.,  0.,  1.],
       [ 1.,  0.,  0.],
       [ 0.,  1.,  0.]])     

# 调整阈值
binarizer = preprocessing.Binarizer(threshold=1.1)
binarizer.transform(X)
#out：
array([[ 0.,  0.,  1.],
       [ 1.,  0.,  0.],
       [ 0.,  0.,  0.]])
```
注：稀疏数据输入：

binarize 和 Binarizer 既接受稠密数据（dense array-like），也接受稀疏矩阵（from scipy.sparse）作为输入

稀疏数据需要转换成压缩的稀疏行（Compressed Sparse Rows）格式（详见scipy.sparse.csr_matrix），为了避免不必要的内存拷贝，推荐使用CSR


## 五）对类别特征进行编码

我们经常会遇到一些类别特征，这些特征不是离散型的数值，而是这样的：["男性","女性"],["来自欧洲","来自美国","来自亚洲"],["使用Firefox浏览器","使用Chrome浏览器","使用Safari浏览器","使用IE浏览器"]等等。这种类型的特征可以被编码为整型（int），如["男性","来自美国","使用IE浏览器"]可以表示成[0,1,3]，["女性","来自亚洲","使用Chrome浏览器"]可以表示成[1,2,1]。这些整数式的表示不能直接作为sklearn的参数，因为我们需要的是连续型的输入，而且我们通常是有序的翻译这些特征，而不是所有的特征都是有序化的（譬如浏览器就是按人工排的序列）。

 　　将这些类别特征转化成sklearn参数中可以使用的方法是：使用one-of-K或者one-hot编码（独热编码OneHotEncoder）。它可以把每一个有m种类别的特征转化成m中二值特征。举例如下：
 ```py
 enc = preprocessing.OneHotEncoder()
#input
enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])  
OneHotEncoder(categorical_features='all', dtype=<... 'float'>,handle_unknown='error', n_values='auto', sparse=True)
#transform
enc.transform([[0, 1, 3]]).toarray()
#out
array([[ 1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.]])
 ```
默认情况下，特征的类别数量是从数据集里自动判断出来的。当然，你也可以用n_values这个参数。我们刚刚举的例子中有两种性别，三种地名和四种浏览器，当我们fit之后就可以将我们的数据转化为数值了。从结果中来看，第一个数字代表性别([0,1]代表男性，女性），第二个数字代表地名（[0,1,2]代表欧洲、美国、亚洲），最后一个数字代表浏览器（[3,0,1,2]代表四种浏览器）
　　此外，字典格式也可以编码： Loading features from dicts
OneHotEncoder参数：class sklearn.preprocessing.OneHotEncoder(n_values='auto', categorical_features='all', dtype=<class 'float'>, sparse=True, handle_unknown='error')

## 六）缺失值的插补
上篇我们讲了五种方法来解决缺失值的问题，其实sklearn里也有一个工具Imputer可以对缺失值进行插补。Imputer类可以对缺失值进行均值插补、中位数插补或者某行/列出现的频率最高的值进行插补，也可以对不同的缺失值进行编码。并且支持稀疏矩阵。

```py
import numpy as np
from sklearn.preprocessing import Imputer
#用均值插补缺失值
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit([[1, 2], [np.nan, 3], [7, 6]])
Imputer(axis=0, copy=True, missing_values='NaN', strategy='mean', verbose=0)
X = [[np.nan, 2], [6, np.nan], [7, 6]]
print(imp.transform(X))                           
[[ 4.          2.        ]
 [ 6.          3.666...]
 [ 7.          6.        ]]

#对稀疏矩阵进行缺失值插补
import scipy.sparse as sp
X = sp.csc_matrix([[1, 2], [0, 3], [7, 6]])
imp = Imputer(missing_values=0, strategy='mean', axis=0)
imp.fit(X)
Imputer(axis=0, copy=True, missing_values=0, strategy='mean', verbose=0)
X_test = sp.csc_matrix([[0, 2], [6, 0], [7, 6]])
print(imp.transform(X_test))                      
[[ 4.          2.        ]
 [ 6.          3.666...]
 [ 7.          6.        ]]
```
在稀疏矩阵中，缺失值被编码为0存储为矩阵中，这种格式是适合于缺失值比非缺失值多得多的情况。此外，Imputer类也可以用于Pipeline中
举个实例(在用随机森林算法之前先用Imputer类进行处理)：
```py
import numpy as np

from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import cross_val_score

rng = np.random.RandomState(0)

dataset = load_boston()
X_full, y_full = dataset.data, dataset.target
n_samples = X_full.shape[0]
n_features = X_full.shape[1]

# Estimate the score on the entire dataset, with no missing values
estimator = RandomForestRegressor(random_state=0, n_estimators=100)
score = cross_val_score(estimator, X_full, y_full).mean()
print("Score with the entire dataset = %.2f" % score)

# Add missing values in 75% of the lines
missing_rate = 0.75
n_missing_samples = np.floor(n_samples * missing_rate)
missing_samples = np.hstack((np.zeros(n_samples - n_missing_samples,
                                      dtype=np.bool),
                             np.ones(n_missing_samples,
                                     dtype=np.bool)))
rng.shuffle(missing_samples)
missing_features = rng.randint(0, n_features, n_missing_samples)

# Estimate the score without the lines containing missing values
X_filtered = X_full[~missing_samples, :]
y_filtered = y_full[~missing_samples]
estimator = RandomForestRegressor(random_state=0, n_estimators=100)
score = cross_val_score(estimator, X_filtered, y_filtered).mean()
print("Score without the samples containing missing values = %.2f" % score)

# Estimate the score after imputation of the missing values
X_missing = X_full.copy()
X_missing[np.where(missing_samples)[0], missing_features] = 0
y_missing = y_full.copy()
estimator = Pipeline([("imputer", Imputer(missing_values=0,
                                          strategy="mean",
                                          axis=0)),
                      ("forest", RandomForestRegressor(random_state=0,
                                                       n_estimators=100))])
score = cross_val_score(estimator, X_missing, y_missing).mean()
print("Score after imputation of the missing values = %.2f" % score)
```

## 七）生成多项式特征 
在输入数据中增加非线性特征可以有效的提高模型的复杂度。简单且常用的方法就是使用多项式特征（polynomial features),可以得到特征的高阶交叉项：
```py
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
X = np.arange(6).reshape(3, 2)
X                                                 
array([[0, 1],
       [2, 3],
       [4, 5]])
poly = PolynomialFeatures(2)
poly.fit_transform(X)                             
array([[  1.,   0.,   1.,   0.,   0.,   1.],
       [  1.,   2.,   3.,   4.,   6.,   9.],
       [  1.,   4.,   5.,  16.,  20.,  25.]])
```

然而有时候我们只需要特征的交叉项，可以设置interaction_only=True来得到：
```py
X = np.arange(9).reshape(3, 3)
X                                                 
array([[0, 1, 2],
       [3, 4, 5],
       [6, 7, 8]])
poly = PolynomialFeatures(degree=3, interaction_only=True)
poly.fit_transform(X)                             
array([[   1.,    0.,    1.,    2.,    0.,    0.,    2.,    0.],
       [   1.,    3.,    4.,    5.,   12.,   15.,   20.,   60.],
       [   1.,    6.,    7.,    8.,   42.,   48.,   56.,  336.]])
```


## 八）自定义转换

　　如果以上的方法觉得都不够，譬如你想用对数据取对数，可以自己用 FunctionTransformer自定义一个转化器,并且可以在Pipeline中使用
```py
import numpy as np
from sklearn.preprocessing import FunctionTransformer
transformer = FunctionTransformer(np.log1p)#括号内的就是自定义函数
X = np.array([[0, 1], [2, 3]])
transformer.transform(X)
array([[ 0.        ,  0.69314718],
       [ 1.09861229,  1.38629436]])
```

告诉你怎么用：

　　如果你在做一个分类任务时，发现第一主成分与这个不相关，你可以用FunctionTransformer把第一列除去，剩下的列用PCA：

```py

import matplotlib.pyplot as plt
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import FunctionTransformer
# 如果报错ImportError: cannot import name FunctionTransformer，可以使用下面的语句
from sklearn.preprocessing import *


def _generate_vector(shift=0.5, noise=15):
    return np.arange(1000) + (np.random.rand(1000) - shift) * noise


def generate_dataset():
    """
    This dataset is two lines with a slope ~ 1, where one has
    a y offset of ~100
    """
    return np.vstack((
        np.vstack((
            _generate_vector(),
            _generate_vector() + 100,
        )).T,
        np.vstack((
            _generate_vector(),
            _generate_vector(),
        )).T,
    )), np.hstack((np.zeros(1000), np.ones(1000)))


def all_but_first_column(X):
    return X[:, 1:]


def drop_first_component(X, y):
    """
    Create a pipeline with PCA and the column selector and use it to
    transform the dataset.
    """
    pipeline = make_pipeline(
        PCA(), FunctionTransformer(all_but_first_column),
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    pipeline.fit(X_train, y_train)
    return pipeline.transform(X_test), y_test


if __name__ == '__main__':
    X, y = generate_dataset()
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50)
    plt.show()
    X_transformed, y_transformed = drop_first_component(*generate_dataset())
    plt.scatter(
        X_transformed[:, 0],
        np.zeros(len(X_transformed)),
        c=y_transformed,
        s=50,
    )
    plt.show()
```

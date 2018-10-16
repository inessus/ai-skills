几种常用的特征选择方法
## 移除低方差的特征 Removing features with low variance
移除所有方差不满足阈值的特征
默认设置下,它将移除所有方差为0的特征,即那些在所有样本中数值完全相同的特征,假设我们有一个带布尔特征的数据集合,我们要移除那些超过80%的数据都为1或者0的特征,该类变量的方差为Var(x) = p(1-p)  p = 0.8
```
from sklean.feature_selection import VarianceThreshold

```
## 单变量特征选择 Univariante feature selection
单变量特征选择基于单变量的统计检测来选择最佳特征
### 卡方检测, Pearson相关系数
SelectKBest 移除得分前 k 名以外的所有特征。 
用于回归: f_regression 
用于分类: chi2 or f_classif

```
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
iris = load_iris()
X, y = iris.data, iris.target
X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
X_new.shape

```
分类： 
chi2计算非负特征和类之间的卡方统计。 
卡方检验用方差来衡量某个观测频率和理论频率之间差异性的方法 
f_classif计算所提供的样本方差F值。 
回归： 
f_regression单变量线性回归检验。计算p-value当噪音比较小的时候，相关性很强，p-value很低。 
简单地说，p-value就是为了验证假设和实际之间一致性的统计学意义的值，即假设检验。有些地方叫右尾概率，根据卡方值和自由度可以算出一个固定的p-value。

### 互信息和最大信息系数
想把互信息直接用于特征选择其实不是太方便：1、它不属于度量方式，也没有办法归一化，在不同数据及上的结果无法做比较；2、对于连续变量的计算不是很方便（X和Y都是集合，x，y都是离散的取值），通常变量需要先离散化，而互信息的结果对离散化的方式很敏感。 
最大信息系数克服了这两个问题。它首先寻找一种最优的离散化方式，然后把互信息取值转换成一种度量方式，取值区间在[0，1]。minepy提供了MIC功能。
```
from minepy import MINE
m = MINE()
x = np.random.uniform(-1, 1, 10000)
m.compute_score(x, x**2)
print m.mic()
```

###　距离相关系数
距离相关系数是为了克服Pearson相关系数的弱点而生的，即便Pearson相关系数是0，我们也不能断定这两个变量是独立的（有可能是非线性相关）；但如果距离相关系数是0，那么我们就可以说这两个变量是独立的。 
尽管有MIC和距离相关系数在了，但当变量之间的关系接近线性相关的时候，Pearson相关系数仍然是不可替代的。第一、Pearson相关系数计算速度快，这在处理大规模数据的时候很重要。第二、Pearson相关系数的取值区间是[-1，1]，而MIC和距离相关系数都是[0，1]。这个特点使得Pearson相关系数能够表征更丰富的关系，符号表示关系的正负，绝对值能够表示强度。当然，Pearson相关性有效的前提是两个变量的变化关系是单调的。

## 递归特征消除(Recursive feature elimination)
递归特征消除 (RFE)通过递归减少考察的特征集规模来选择特征。 
首先，预测模型在原始特征上训练，每项特征指定一个权重。之后，那些拥有最小绝对值权重的特征被踢出特征集。如此往复递归，直至剩余的特征数量达到所需的特征数量。 
RFE具有递归特征消除的特征排序。 
RFECV 通过交叉验证的方式执行RFE，以此来选择最佳数量的特征。
```
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.feature_selection import RFE
# Load the digits dataset
digits = load_digits()
X = digits.images.reshape((len(digits.images), -1))
y = digits.target
svc = SVC(kernel="linear", C=1)
##estimator模型能表示特征权重的模型，n_features_to_select选择的特征个数 step每次迭代时删除的(整数)特征数 如果在(0.0, 1.0)代表每次删除特征的百分比
rfe = RFE(estimator=svc, n_features_to_select=10, step=1)
rfe.fit(X, y)
#查看每列是否选中
print(rfe.support_)
##查看特征的排序
print(rfe.ranking_)
##输出选择后的特征
def featureTop(feature_array,rank_array,TopK):
    indexTop=rank_array[0:TopK]
    re=[]
    for i in range(feature_array.shape[0]):
        t=feature_array[i][list(indexTop)]
        re.append(list(t))
    return np.array(re)
##进行转换
tm=featureTop(X,rfe.ranking_,10)
print(tm)
```
```
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.feature_selection import RFECV
from sklearn.cross_validation import StratifiedKFold
# Load the digits dataset
digits = load_digits()
X = digits.images.reshape((len(digits.images), -1))
y = digits.target
svc = SVC(kernel="linear", C=1)
##estimator模型，step每次迭代时删除的（整数）特征数，cv交叉验证(可为整数cv=3)，scoring评估函数
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(y, 2), scoring='accuracy')
rfecv.fit(X, y)
#查看每列是否选中
rfecv.support_ 
##查看特征的排序
rfecv.ranking_
##交叉验证的成绩
rfecv.grid_scores_ 
```

## 使用模型选择特征(Feature selection using SelectFromModel)
### 基于L1的特征选择(L1-based feature selection)
对于SVM和逻辑回归，参数C控制稀疏性：C越小，被选中的特征越少。对于Lasso，参数alpha越大，被选中的特征越少 。 
常用于此目的的稀疏预测模型有 linear_model.Lasso （回归）， linear_model.LogisticRegression 和 svm.LinearSVC （分类） 
回归查看分数lasso.coef_
```
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
iris = load_iris()
X, y = iris.data, iris.target
X.shape
##(150, 4)
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(X)
X_new.shape
```

### 随机稀疏模型(Randomized sparse models)
基于L1的稀疏模型的局限在于，当面对一组互相关的特征时，它们只会选择其中一项特征。为了减轻该问题的影响可以使用随机化技术，通过多次重新估计稀疏模型来扰乱设计矩阵，或通过多次下采样数据来统计一个给定的回归量被选中的次数。 
RandomizedLasso 实现了使用这项策略的Lasso 
RandomizedLogisticRegression 使用逻辑回归，适用于分类 
它的主要思想是在不同的数据子集和特征子集上运行特征选择算法，不断的重复，最终汇总特征选择结果，比如可以统计某个特征被认为是重要特征的频率（被选为重要特征的次数除以它所在的子集被测试的次数）。理想情况下，重要特征的得分会接近100%。稍微弱一点的特征得分会是非0的数，而最无用的特征得分将会接近于0。
```
from sklearn.linear_model import RandomizedLogisticRegression
randomized_logistic = RandomizedLogisticRegression().fit(X, y)
randomized_logistic.scores_
```

### 基于树的特征选择(Tree-based feature selection)
基于树的预测模型能够用来计算特征的重要程度

```
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
iris = load_iris()
X, y = iris.data, iris.target
X.shape
##(150, 4)
clf = ExtraTreesClassifier()
##clf =GradientBoostingClassifier()
clf = clf.fit(X, y)
clf.feature_importances_  
##array([ 0.04...,  0.05...,  0.4...,  0.4...])
model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X)
X_new.shape  
```

## 特征选择融入pipeline
我们将 sklearn.svm.LinearSVC 和 sklearn.feature_selection.SelectFromModel 结合来评估特征的重要性，并选择最相关的特征。之后 sklearn.ensemble.RandomForestClassifier 模型使用转换后的输出训练，即只使用被选出的相关特征。

```
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
clf = Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1"))),
  ('classification', RandomForestClassifier())
])
clf.fit(X, y)
```
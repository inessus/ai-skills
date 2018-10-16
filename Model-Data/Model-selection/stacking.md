集成学习中，bagging和boosting可以直接调用，而stacking则需要自己设计调配一下才行，尤其是交叉验证的环节比较麻烦。
之前使用了blending（stacking）融合了多个推荐系统模型，性能提升很小,后来无意中发现了mlxtend这个工具包，里面集成了stacking分类回归模型以及它们的交叉验证的版本。
一般推荐系统中的模型融合会称为blending而不是stacking，这是netfilx price比赛中的论文形成的习惯，但实际上推荐系统中blending融合和stacking没有什么分别，现在一般认为blending的融合是弱化版的stacking，是切分样本集为不相交的子样本然后用各个算法生成结果再融合，这种方法不能够最大限度的利用数据，而stacking是得到各个算法训练全样本的结果再用一个元算法融合这些结果，效果会比较好一些，它可以选择使用网格搜索和交叉验证。

stacking回归是一种通过元回归器组合多个回归模型的集成学习技术。每个独立的基回归模型在训练时都要使用整个训练集；那么，在集成学习过程中独立的基回归模型的输出作为元特征成为元回归器的输入，元回归器通过拟合这些元特征来组合多个模型。

## 2 Example 1 - Simple Stacked Regression
```
from mlxtend.regressor import StackingRegressor
from mlxtend.data import boston_housing_data
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np

%matplotlib inline

# 生成一个样本数据集
np.random.seed(1)
X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - np.random.rand(8))

# 初始化模型
lr = LinearRegression()
svr_lin = SVR(kernel='linear')
ridge = Ridge(random_state=1)
svr_rbf = SVR(kernel='rbf')
#融合四个模型
stregr = StackingRegressor(regressors=[svr_lin, lr, ridge], meta_regressor=svr_rbf)

# 训练stacking分类器
stregr.fit(X, y)
stregr.predict(X)

# 拟合结果的评估和可视化
print("Mean Squared Error: %.4f" % np.mean((stregr.predict(X) - y) ** 2))
print('Variance Score: %.4f' % stregr.score(X, y))
with plt.style.context(('seaborn-whitegrid')):
    plt.scatter(X, y, c='lightgray')
    plt.plot(X, stregr.predict(X), c='darkgreen', lw=2)

plt.show()
```

## 3 Example 2 - Stacked Regression and GridSearch
为了给sklearn-learn中的网格搜索设置参数网格，我们在参数网格中提供了学习器的名字，对于元回归器的情况，我们加了“meta-”前缀。
```
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

# 初始化模型
lr = LinearRegression()
svr_lin = SVR(kernel='linear')
ridge = Ridge(random_state=1)
lasso = Lasso(random_state=1)
svr_rbf = SVR(kernel='rbf')
regressors = [svr_lin, lr, ridge, lasso]
stregr = StackingRegressor(regressors=regressors,meta_regressor=svr_rbf)

params = {'lasso__alpha': [0.1, 1.0, 10.0],
          'ridge__alpha': [0.1, 1.0, 10.0],
                'svr__C': [0.1, 1.0, 10.0],
           'meta-svr__C': [0.1, 1.0, 10.0, 100.0],
       'meta-svr__gamma': [0.1, 1.0, 10.0]}

grid = GridSearchCV(estimator=stregr,
                    param_grid=params,
                    cv=5,
                    refit=True)

grid.fit(X, y)
for params, mean_score, scores in grid.grid_scores_:
    print("%0.3f +/- %0.2f %r"
        % (mean_score, scores.std() / 2.0, params))

# 拟合结果的评估和可视化
print("Mean Squared Error: %.4f"
% np.mean((grid.predict(X) - y) ** 2))
print('Variance Score: %.4f' % grid.score(X, y))
with plt.style.context(('seaborn-whitegrid')):
    plt.scatter(X, y, c='lightgray')
    plt.plot(X, grid.predict(X), c='darkgreen', lw=2)
plt.show()
```
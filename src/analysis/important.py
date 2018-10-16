import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import warnings
import lightgbm as lgb


from tqdm import tqdm
from scipy.stats import ks_2samp
from wordcloud import WordCloud
from sklearn import model_selection, preprocessing, metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
import itertools
from sklearn.metrics import confusion_matrix


def important_variables_xgb(train_X, train_y):
    xgb_params = {
        'eta': 0.05,
        'max_depth': 6,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'objective': 'reg:linear',
        'silent': 1
    }
    dtrain = xgb.DMatrix(train_X, train_y, feature_names=train_X.columns.values)
    # model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100, feval=xgb_r2_score, maximize=True)
    model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100, maximize=True)

    # plot the important features #
    fig, ax = plt.subplots(figsize=(12, 18))
    xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
    plt.show()


def important_variables_ensemble(train_X, train_y):
    from sklearn import ensemble
    train_X.fillna(0, inplace=True)
    model = ensemble.RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=4, max_features=0.2, n_jobs=-1, random_state=0)
    model.fit(train_X, train_y)
    feat_names = train_X.columns.values

    # plot the importances
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    indices = np.argsort(importances)[::-1][:20]

    plt.figure(figsize=(12,12))
    plt.title("Feature importances")
    plt.bar(range(len(indices)), importances[indices], color="r", align="center")
    plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical')
    plt.xlim([-1, len(indices)])
    plt.show()


def important_variables_by_ET(train_df, target_y, exclude=[], create_plot=True):
    if len(exclude) > 0:
        train_df = train_df.drop(exclude, axis=1)
    feat_names = train_df.columns.values

    from sklearn import ensemble
    model = ensemble.ExtraTreesRegressor(n_estimators=30, max_depth=20, max_features=0.3, n_jobs=-1, random_state=0)
    model.fit(train_df, train_y)

    ## plot the importances ##
    if create_plot:
        importances = model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
        indices = np.argsort(importances)[::-1][:20]
        
        plt.figure(figsize=(12,12))
        plt.title("Feature importances")
        plt.bar(range(len(indices)), importances[indices], color="r", yerr=std[indices], align="center")
        plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical')
        plt.xlim([-1, len(indices)])
        plt.show()   


def important_variables_by_XGB(train_df, target, exclude=[], create_plot=True):
    train_y = train_df[target].values
    if len(exclude) > 0:
        train_df = train_df.drop(exclude, axis=1)
    import xgboost as xgb
    xgb_params = {
        'eta': 0.05,
        'max_depth': 8,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'objective': 'reg:linear',
        'silent': 1,
        'seed': 0
    }
    dtrain = xgb.DMatrix(train_df, train_y, feature_names=train_df.columns.values)
    model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=50)

    # plot the important features #
    if create_plot:
        fig, ax = plt.subplots(figsize=(12, 18))
        xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
        plt.show()


def test_prediction(data, train_idx):
    """
        Try to classify train/test samples from total dataframe
        # Run classification on total raw data
        test_prediction(total_df)
        :param data:
        :param train_idx:
        :return:
    """

    # Create a target which is 1 for training rows, 0 for test rows
    y = np.zeros(len(data))
    y[train_idx] = 1

    # Perform shuffled CV predictions of train/test label
    predictions = cross_val_predict(
        ExtraTreesClassifier(n_estimators=100, n_jobs=4),
        data, y,
        cv=StratifiedKFold(
            n_splits=10,
            shuffle=True,
            random_state=42
        )
    )

    # Show the classification report
    print(classification_report(y, predictions))


def get_diff_columns(train_df, test_df, show_plots=True, show_all=False, threshold=0.1):
    """
        # Get the columns which differ a lot between test and train
        diff_df = get_diff_columns(total_df.iloc[train_idx], total_df.iloc[test_idx])
        Use KS to estimate columns where distributions differ a lot from each other

        :param train_df:
        :param test_df:
        :param show_plots:
        :param show_all:
        :param threshold:
        :return:
    """

    # Find the columns where the distributions are very different
    train_df = train_df.select_dtypes(include=[np.number])
    test_df = test_df.select_dtypes(include=[np.number])
    diff_data = []
    for col in tqdm(train_df.columns):
        statistic, pvalue = ks_2samp(
            train_df[col].values,
            test_df[col].values
        )
        if pvalue <= 0.05 and np.abs(statistic) > threshold:
            diff_data.append({'prepare': col, 'p': np.round(pvalue, 5), 'statistic': np.round(np.abs(statistic), 2)})
        # diff_data.append({'prepare': col, 'p': np.round(pvalue, 5), 'statistic': np.round(np.abs(statistic), 2)})

    # Put the differences into a dataframe
    diff_df = pd.DataFrame(diff_data).sort_values(by='statistic', ascending=False)
    print(diff_df.shape)

    if show_plots:
        # Let us see the distributions of these columns to confirm they are indeed different
        n_cols = 5
        if show_all:
            n_rows = int(len(diff_df) / 5)
        else:
            n_rows = 2
        _, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3 * n_rows))
        axes = [x for l in axes for x in l]

        # Create plots
        for i, (_, row) in enumerate(diff_df.iterrows()):
            if i >= len(axes):
                break
            extreme = np.max(np.abs(train_df[row.feature].tolist() + test_df[row.feature].tolist()))
            train_df.loc[:, row.feature].apply(np.log1p).hist(
                ax=axes[i], alpha=0.5, label='Train', density=True,
                bins=np.arange(-extreme, extreme, 0.25)
            )
            test_df.loc[:, row.feature].apply(np.log1p).hist(
                ax=axes[i], alpha=0.5, label='Test', density=True,
                bins=np.arange(-extreme, extreme, 0.25)
            )
            axes[i].set_title(f"Statistic = {row.statistic}, p = {row.p}")
            axes[i].set_xlabel(f'Log({row.prepare})')
            axes[i].legend()

        plt.tight_layout()
        plt.show()

    return diff_df


def run_lgb(train_X, train_y, val_X, val_y, test_X, create_plot=False):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 30,
        "learning_rate": 0.1,
        "bagging_fraction": 0.7,
        "feature_fraction": 0.7,
        "bagging_frequency": 5,
        "bagging_seed": 2018,
        "verbosity": -1
    }

    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    evals_result = {}
    model = lgb.train(params, lgtrain, 1000, valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=20,
                      evals_result=evals_result)

    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)

    if create_plot:
        fig, ax = plt.subplots(figsize=(12, 18))
        lgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
        ax.grid(False)
        plt.title("LightGBM - Feature Importance", fontsize=15)
        plt.show()

    return pred_test_y, model, evals_result


def cross_validation_KFold(train_X, train_y, test_X, runLGB):
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
    pred_test_full = 0
    for dev_index, val_index in kf.split(train_X):
        dev_X, val_X = train_X.loc[dev_index, :], train_X.loc[val_index, :]
        dev_y, val_y = train_y[dev_index], train_y[val_index]
        pred_test, model, evals_result = runLGB(dev_X, dev_y, val_X, val_y, test_X)
        pred_test_full += pred_test
    pred_test_full /= 5.
    pred_test_full = np.expm1(pred_test_full)


def cross_validation_KFold_V1(train_X, train_y, test_X, runXGB):
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros([train_X.shape[0], 3])
    for dev_index, val_index in kf.split(train_X):
        dev_X, val_X = train_X.loc[dev_index], train_X.loc[val_index]
        dev_y, val_y = train_y[dev_index], train_y[val_index]
        pred_val_y, pred_test_y, model = runXGB(dev_X, dev_y, val_X, val_y, test_X, seed_val=0)
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index,:] = pred_val_y
        cv_scores.append(metrics.log_loss(val_y, pred_val_y))
        break
    print("cv scores : ", cv_scores)


# Function to create confusion matrix ###
# From http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py #
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    #else:
    #    print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_matrix(val_y, pred_val_y):
    cnf_matrix = confusion_matrix(val_y, np.argmax(pred_val_y,axis=1))
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure(figsize=(8,8))
    plot_confusion_matrix(cnf_matrix, classes=['EAP', 'HPL', 'MWS'],
                          title='Confusion matrix, without normalization')
    plt.show()

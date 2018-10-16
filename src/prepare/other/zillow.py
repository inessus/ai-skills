import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

color = sns.color_palette()

# %matplotlib inline
warnings.filterwarnings("ignore")  

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999


def read_data():
    train_df = pd.read_csv('../input/train_2016_v2.csv', parse_dates=['transactiondate'])
    return train_df


def target_sort_scatter_plot(df, target):
    plt.figure(figsize=(8, 6))
    plt.scatter(range(df.shape[0]), np.sort(df[target].values))
    plt.xlabel('index', fontsize=12)
    plt.ylabel(target, fontsize=12)
    plt.show()


def outlier_dist_plot(df, col, ul=99, ll=1, create_plot=True):
    ulimit = np.percentile(df[col].values, ul)
    llimit = np.percentile(df[col].values, ll)
    df[col].loc[df[col] > ulimit] = ulimit
    df[col].loc[df[col] < llimit] = llimit
    
    if create_plot:
        plt.figure(figsize=(12, 8))
        sns.distplot(df[col].values, bins=50, kde=False)
        plt.xlabel(col, fontsize=12)
        plt.show()


def extract_feature_from_date(df, col, create_plot=True):
    df['transaction_month'] = df[col].dt.month
    cnt_srs = df['transaction_month'].value_counts()
    color = sns.color_palette()
    if create_plot:
        plt.figure(figsize=(12, 6))
        sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[3])
        plt.xticks(rotation='vertical')
        plt.xlabel('Month of transactions', fontsize=12)
        plt.xlabel('Number of Occurences', fontsize=12)
        plt.show()


def read_prop():
    df = pd.read_csv('../input/properties_2016.csv')
    return df


def plot_null_values(df, create_plot=True):
    missing_df = df.isnull().sum(axis=0).reset_index()
    missing_df.columns = ['column_name', 'missing_count']
    missing_df = missing_df.loc[missing_df['missing_count'] > 0]
    missing_df = missing_df.sort_values(by='missing_count')

    if create_plot:
        ind = np.arange(missing_df.shape[0])
        width = 0.9
        fig, ax = plt.subplots(figsize=(12, 18))
        rects = ax.barh(ind, missing_df.missing_count.values, color='blue')
        ax.set_yticks(ind)
        ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
        ax.set_xlabel("Count of missing values")
        ax.set_title("Number of missing values in each column")
        plt.show()    


def join_plot(df, col1, col2):
    plt.figure(figsize=(12, 12))
    sns.jointplot(x=df[col1].values, y=df[col2].values, size=10)
    plt.ylabel(col1, fontsize=12)
    plt.ylabel(col2, fontsize=12)
    plt.show()


def merge(train, prop):
    return pd.merge(train, prop, on='parcelid', how='left')


def show_dtypes(df):
    pd.options.display.max_rows = 65
    dtype_df = df.dtypes.reset_index()
    dtype_df.columns = ['Count', 'Column Type']
    print(dtype_df)
    print(dtype_df.groupby("Column Type").aggregate('count').reset_index())


def show_missing_ratio(df, limit=0.999):
    missing_df = df.isnull().sum(axis=0).reset_index()
    missing_df.columns = ['column_name', 'missing_count']
    missing_df['missing_ratio'] = missing_df['missing_count'] / df.shape[0]
    print(missing_df.loc[missing_df['missing_ratio'] > limit])


def univariate_analysis(df, target, create_plot=True):
    # Let us just impute the missing values with mean values to compute correlation coefficients #
    mean_values = df.mean(axis=0)
    df.fillna(mean_values, inplace=True)

    # Now let us look at the correlation coefficient of each of these variables
    x_cols = [col for col in df.columns if col not in [target] if df[col].dtype=='float64']
    
    labels = []
    values = []
    for col in x_cols:
        labels.append(col)
        values.append(np.corrcoef(df[col].values, df[target].values)[0, 1])
    corr_df = pd.DataFrame({'col_labels':labels, 'corr_values':values})
    corr_df = corr_df.sort_values(by='corr_values')
    
    if create_plot:
        ind = np.arange(len(labels))
        width = 0.9
        fig, ax = plt.subplots(figsize=(12, 40))
        rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='y')
        ax.set_yticks(ind)
        ax.set_yticklabels(corr_df.col_labels.values, rotation='horizontal')
        ax.set_xlabel('Correlation coefficient')
        ax.set_ylabel('Correlation coefficient of the variables')
        plt.show()

    # There are few variables at the top of this graph without any correlation values. 
    # I guess they have only one unique value and hence no correlation value. Let us confirm the same.
    corr_zero_cols = ['assessmentyear', 'storytypeid', 'pooltypeid2', 'pooltypeid7', 'pooltypeid10', 'poolcnt', 'decktypeid', 'buildingclasstypeid']
    for col in corr_zero_cols:
        print(col, len(df[col].unique()))
        
    if create_plot:
        # The important variables themselves are very highly correlated.! Let us now look at each of them.
        corr_df_sel = corr_df.loc[(corr_df['corr_values']>0.02) | (corr_df['corr_values'] < -0.01)]         
        cols_to_use = corr_df_sel.col_labels.tolist()
        print(cols_to_use)
        temp_df = df[cols_to_use]
        corrmat = temp_df.corr(method='spearman')
        f, ax = plt.subplots(figsize=(8, 8))

        # Draw the heatmap using seaborn
        sns.heatmap(corrmat, vmax=1., square=True)
        plt.title("Important variables correlation map", fontsize=15)
        plt.show()


def trim_col_vs_target_joint_plot(df, col, target, log1p=False, ul=99.5, ll=0.5, create_plot=True):
    ulimit = np.percentile(df[col].values, ul)
    llimit = np.percentile(df[col].values, ll)
    df[col].loc[df[col] > ulimit] = ulimit
    df[col].loc[df[col] < llimit] = llimit
    color = sns.color_palette()

    if create_plot:
        plt.figure(figsize=(12, 12))
        if log1p:
            sns.jointplot(x=np.log1p(df[col].values), y=np.log1p(df[target].values), size=10, color=color[4])
        else:
            sns.jointplot(x=df[col].values, y=df[target].values, size=10, color=color[4])
        plt.ylabel(target, fontsize=12)
        plt.xlabel(col, fontsize=12)
        plt.title(col + ' VS ' + target, fontsize=12)
        plt.show()


def col_count_plot(df, col, create_plot=True):
    plt.figure(figsize=(12, 8))
    sns.countplot(x=col, data=df)
    plt.ylabel('Count', fontsize=12)
    plt.xlabel(col, fontsize=12)
    plt.xticks(rotation='vertical')
    plt.title('Frequency of ' + col + " Count ", fontsize=15)
    plt.show()


def important_variables_by_ET(train_df, target, exclude=[], create_plot=True):
    train_y = train_df[target].values
    if len(exclude) > 0:
        train_df = train_df.drop(exclude, axis=1)
    feat_names = train_df.columns.values

    from sklearn import ensemble
    model = ensemble.ExtraTreesRegressor(n_estimators=25, max_depth=30, max_features=0.3, n_jobs=-1, random_state=0)
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
    train_y = train_df['logerror'].values
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
        'seed' : 0
    }
    dtrain = xgb.DMatrix(train_df, train_y, feature_names=train_df.columns.values)
    model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=50)

    # plot the important features #
    if create_plot:
        fig, ax = plt.subplots(figsize=(12,18))
        xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
        plt.show()


if __name__ == "__main__":   
    
    train_df = read_data()
    target_plot(train_df, 'logerror')
    outlier_target(train_df, 'logerror')
    extract_feature_from_date(train_df, 'transactiondate')
    prop_df = read_prop()
    prop_df.head()
    plot_null_values(prop_df)
    join_plot(prop_df, 'longitude', 'latitude')
    train_df_all = merge(train_df, prop_df)
    show_dtypes(train_df_all)
    show_missing_ratio(train_df_all)
    univariate_analysis(train_df_all, 'logerror')
    trim_col_vs_target_joint_plot(train_df, "finishedsquarefeet12", 'logerror')
    trim_col_vs_target_joint_plot(train_df, "calculatedfinishedsquarefeet", 'logerror')
    col_count_plot(train_df, 'bathroomcnt')  
    trim_col_vs_target_joint_plot(train_df, 'taxamount', 'logerror')
    exclude = ["hashottuborspa", "propertycountylandusecode", "propertyzoningdesc", "fireplaceflag", "taxdelinquencyflag"] \
        + ['parcelid', 'logerror', 'transactiondate']
    important_variables_by_ET(train_df, 'logerror',exclude)
    important_variables_by_XGB(train_df, 'logerror', exclude)   
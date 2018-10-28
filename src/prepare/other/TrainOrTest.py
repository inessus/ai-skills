import gc
import itertools
from copy import deepcopy

import numpy as np
import pandas as pd

from tqdm import tqdm

from scipy.stats import ks_2samp

from sklearn.preprocessing import scale, MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection


from sklearn import manifold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

import warnings

warnings.filterwarnings('ignore')


def combined_data(train, test):
    """
        Get the combined data
        :param trainer pandas.dataframe:
        :param test pandas.dataframe:
        :return pandas.dataframe:
    """
    A = set(train.columns.values)
    B = set(test.columns.values)
    colToDel = A.difference(B)
    total_df = pd.concat([train.drop(colToDel, axis=1), test], axis=0)
    return total_df


def remove_duplicate_columns(total_df):
    """
        Removing duplicate columns
    """
    colsToRemove = []
    columns = total_df.columns
    for i in range(len(columns) - 1):
        v = total_df[columns[i]].values
        for j in range(i + 1, len(columns)):
            if np.array_equal(v, total_df[columns[j]].values):
                colsToRemove.append(columns[j])
    colsToRemove = list(set(colsToRemove))
    total_df.drop(colsToRemove, axis=1, inplace=True)
    print(f">> Dropped {len(colsToRemove)} duplicate columns")
    return total_df


def log_significant_outliers(total_df):
    """
        frist master fill na
        Log-transform all columns which have significant outliers (> 3x standard deviation)
    :return pandas.dataframe:
    """
    total_df_all = deepcopy(total_df).select_dtypes(include=[np.number])
    total_df_all.fillna(0, inplace=True)  # ********
    for col in total_df_all.columns:
        # print(col)
        data = total_df_all[col].values
        data_mean, data_std = np.mean(data), np.std(data)
        cut_off = data_std * 3
        lower, upper = data_mean - cut_off, data_mean + cut_off
        outliers = [x for x in data if x < lower or x > upper]

        if len(outliers) > 0:
            non_zero_index = data != 0
            total_df_all.loc[non_zero_index, col] = np.log(data[non_zero_index])

        non_zero_rows = total_df[col] != 0
        total_df_all.loc[non_zero_rows, col] = scale(total_df_all.loc[non_zero_rows, col])
        gc.collect()

    return total_df_all


def test_pca(data, train_idx, test_idx, create_plots=True):
    """
        data, panda.DataFrame
        train_idx = range(0, len(train_df))
        test_idx = range(len(train_df), len(total_df))
        Run PCA analysis, return embeding
    """
    data = data.select_dtypes(include=[np.number])
    data = data.fillna(0)
    # Create a PCA object, specifying how many components we wish to keep
    pca = PCA(n_components=len(data.columns))

    # Run PCA on scaled numeric dataframe, and retrieve the projected data
    pca_trafo = pca.fit_transform(data)

    # The transformed data is in a numpy matrix. This may be inconvenient if we want to further
    # process the data, and have a more visual impression of what each column is etc. We therefore
    # put transformed/projected data into new dataframe, where we specify column names and index
    pca_df = pd.DataFrame(
        pca_trafo,
        index=data.index,
        columns=['PC' + str(i + 1) for i in range(pca_trafo.shape[1])]
    )

    if create_plots:
        # Create two plots next to each other
        _, axes = plt.subplots(2, 2, figsize=(20, 15))
        axes = list(itertools.chain.from_iterable(axes))

        # Plot the explained variance# Plot t
        axes[0].plot(
            pca.explained_variance_ratio_, "--o", linewidth=2,
            label="Explained variance ratio"
        )

        # Plot the explained variance# Plot t
        axes[0].plot(
            pca.explained_variance_ratio_.cumsum(), "--o", linewidth=2,
            label="Cumulative explained variance ratio"
        )

        # show legend
        axes[0].legend(loc='best', frameon=True)

        # show biplots
        for i in range(1, 4):
            # Components to be plottet
            x, y = "PC" + str(i), "PC" + str(i + 1)

            # plot biplots
            settings = {'kind': 'scatter', 'ax': axes[i], 'alpha': 0.2, 'x': x, 'y': y}

            pca_df.iloc[train_idx].plot(label='Train', c='#ff7f0e', **settings)
            pca_df.iloc[test_idx].plot(label='Test', c='#1f77b4', **settings)
    return pca_df


# tsne_df = test_tsne(pca_df, train_idx, test_idx,title='t-SNE: Scaling on non-zeros')
def test_tsne(data, train_idx, test_idx, title='t-SNE', create_plots=True):
    """
        Run t-SNE return embedind
        train_idx = range(0, len(train_df))
        test_idx = range(len(train_df), len(total_df))
    """

    # run t-SNE
    tsne = TSNE(n_components=2, init='pca')
    Y = tsne.fit_transform(data)

    # Create plot
    # Run t-SNE on PCA embedding
    if create_plots:
        _, axes = plt.subplots(1, 1, figsize=(10, 8))
        for name, idx in zip(['Train', 'Test'], [train_idx, test_idx]):
            axes.scatter(Y[idx, 0], Y[idx, 1], label=name, alpha=0.2)
            axes.set_title(title)
            axes.xaxis.set_major_formatter(NullFormatter())
            axes.yaxis.set_major_formatter(NullFormatter())
        axes.legend()
        plt.axis('tight')
        plt.show()
    return Y


def color_plot_sne(tsne_df):
    """
        t-SNE color plot
        :param tsne_df:
        :return:
    """
    gc.collect()
    # Get our color map
    cm = plt.cm.get_cmap('RdYlBu')

    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    sc = axes[0].scatter(tsne_df[:, 0], tsne_df[:, 1], alpha=0.2, c=range(len(tsne_df)), cmap=cm)
    cbar = fig.colorbar(sc, ax=axes[0])
    cbar.set_label('Entry index')
    axes[0].set_title("t-SNE colored by index")
    axes[0].xaxis.set_major_formatter(NullFormatter())
    axes[0].yaxis.set_major_formatter(NullFormatter())

    zero_count = (tsne_df == 0).sum(axis=1)
    sc = axes[1].scatter(tsne_df[:, 0], tsne_df[:, 1], alpha=0.2, c=zero_count, cmap=cm)
    cbar = fig.colorbar(sc, ax=axes[1])
    cbar.set_label('#sparse entries')
    axes[1].set_title("t-SNE colored by number of zeros")
    axes[1].xaxis.set_major_formatter(NullFormatter())
    axes[1].yaxis.set_major_formatter(NullFormatter())


def target_color_plot(tsne_df, target_col, train_idx):
    """
        target_col = train_df['Survived']
        target_color_plot(tsne_df, target_col)
        :param tsne_df:
        :param target_col:
        :param train_idx:
        :return:
    """
    # Create plot
    cm = plt.cm.get_cmap('RdYlBu')
    fig, axes = plt.subplots(1, 1, figsize=(10, 8))
    sc = axes.scatter(tsne_df[train_idx, 0], tsne_df[train_idx, 1], alpha=0.2, c=np.log1p(target_col), cmap=cm)
    cbar = fig.colorbar(sc, ax=axes)
    cbar.set_label('Log1p(target)')
    axes.set_title("t-SNE colored by target")
    axes.xaxis.set_major_formatter(NullFormatter())
    axes.yaxis.set_major_formatter(NullFormatter())


def perplexity_tsne_plot(pca_df, train_idx, test_idx):
    """
    t-SNE can give some pretty tricky to intepret results depending on the perplexity parameter used.
    So just to be sure in the following I check for a few different values of the perplexity parameter.
    :param pca_df:
    :param train_idx:
    :param test_idx:
    :return:
    """
    _, axes = plt.subplots(1, 4, figsize=(20, 5))
    for i, perplexity in enumerate([5, 30, 50, 100]):

        # Create projection
        Y = TSNE(init='pca', perplexity=perplexity).fit_transform(pca_df)

        # Plot t-SNE
        for name, idx in zip(["Train", "Test"], [train_idx, test_idx]):
            axes[i].scatter(Y[idx, 0], Y[idx, 1], label=name, alpha=0.2)
        axes[i].set_title("Perplexity=%d" % perplexity)
        axes[i].xaxis.set_major_formatter(NullFormatter())
        axes[i].yaxis.set_major_formatter(NullFormatter())
        axes[i].legend()

    plt.show()


def test_prediction(data, train_idx):
    """
        Try to classify trainer/test samples from total dataframe
        # Run classification on total raw data
        test_prediction(total_df)
        :param data:
        :param train_idx:
        :return:
    """

    # Create a target which is 1 for training rows, 0 for test rows
    y = np.zeros(len(data))
    y[train_idx] = 1

    # Perform shuffled CV predictions of trainer/test label
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
        # Get the columns which differ a lot between test and trainer
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


def all_method(total_df, train_idx, test_idx, target_col):
    """

    :param total_df:
    :param train_idx:
    :param test_idx:
    :param target_col: target_col=train_df['Survived'])
    :return:
    """
    COMPONENTS = 5

    # List of decomposition methods to use
    methods = [
        TruncatedSVD(n_components=COMPONENTS),
        PCA(n_components=COMPONENTS),
        FastICA(n_components=COMPONENTS),
        GaussianRandomProjection(n_components=COMPONENTS, eps=0.1),
        SparseRandomProjection(n_components=COMPONENTS, dense_output=True)
    ]

    # Run all the methods
    embeddings = []
    for method in methods:
        name = method.__class__.__name__
        embeddings.append(
            pd.DataFrame(method.fit_transform(total_df), columns=[f"{name}_{i}" for i in range(COMPONENTS)])
        )
        print(f">> Ran {name}")

    # Put all components into one dataframe
    components_df = pd.concat(embeddings, axis=1)

    # Prepare plot
    _, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Run t-SNE on components
    tsne_df = test_tsne(
        components_df, train_idx, test_idx,
        title='t-SNE: with decomposition features'
    )

    # Color by index
    fig, axes = plt.subplots(2, 1, figsize=(20, 20))
    cm = plt.cm.get_cmap('RdYlBu')
    sc = axes[0].scatter(tsne_df[:, 0], tsne_df[:, 1], alpha=0.2, c=range(len(tsne_df)), cmap=cm)
    cbar = fig.colorbar(sc, ax=axes[0])
    cbar.set_label('Entry index')
    axes[0].set_title("t-SNE colored by index")
    axes[0].xaxis.set_major_formatter(NullFormatter())
    axes[0].yaxis.set_major_formatter(NullFormatter())

    # Color by target

    sc = axes[1].scatter(tsne_df[train_idx, 0], tsne_df[train_idx, 1], alpha=0.2, c=np.log1p(target_col), cmap=cm)
    cbar = fig.colorbar(sc, ax=axes[1])
    cbar.set_label('Log1p(target)')
    axes[1].set_title("t-SNE colored by target")
    axes[1].xaxis.set_major_formatter(NullFormatter())
    axes[1].yaxis.set_major_formatter(NullFormatter())

    plt.axis('tight')
    plt.show()
    return components_df


if __name__ == '__main__':
    print('test')
    SAMPLE = 4459
    train_df = pd.read_csv('/home/webdev/ai/kaggle/trainortest/trainer.csv').sample(SAMPLE)
    test_df = pd.read_csv('/home/webdev/ai/kaggle/trainortest/test.csv').sample(SAMPLE)

    total_df_all = combined_data(train_df, test_df)
    print(">> Combined:", total_df_all.shape)
    total_df_all = remove_duplicate_columns(total_df_all)
    print(">> Remove duplicate:", total_df_all.shape)
    total_df = log_significant_outliers(total_df_all)
    print(">> Log outliers : (only for np.number)", total_df.shape)
    train_idx = range(0, len(train_df))
    test_idx = range(len(train_df), len(total_df))

    pca_df = test_pca(total_df, train_idx, test_idx)
    pca_df_all = test_pca(total_df_all, train_idx, test_idx)
    print(">> PCA : (only for np.number)", pca_df.shape, pca_df_all.shape)

    tsne_df = test_tsne(pca_df, train_idx, test_idx, title='t-SNE: Scaling on non-zeros')
    print(">> TSNE : (only for np.number)", tsne_df.shape)

    color_plot_sne(tsne_df)
    print(">> TSNE  color: (only for np.number)", tsne_df.shape)

    print('>> Prediction Train or Test')
    test_prediction(total_df, train_idx)

    diff_df = get_diff_columns(total_df.iloc[train_idx], total_df.iloc[test_idx])
    print(">> Diff columns")

    print(f">> Dropping {len(diff_df)} features based on KS tests")
    test_prediction(total_df.drop(diff_df.feature.values, axis=1), train_idx)

    # target = train_df['Survived']
    target = train_df['target']
    all_method(total_df, train_idx, test_idx, target)

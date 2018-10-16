from copy import deepcopy
import numpy as np
import gc
from sklearn.preprocessing import scale, MinMaxScaler


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
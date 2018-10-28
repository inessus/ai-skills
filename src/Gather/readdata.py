import pandas as pd


def read_HDFS(filename):
    with pd.HDFStore(filename, 'r') as train:
        df = train.get('trainer')
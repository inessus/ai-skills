""" CNN/DM dataset"""
import json
import re
import os
from os.path import join
import pandas as pd
import platform

from torch.utils.data import Dataset


class JsonFileDataset(Dataset):
    """
        train 训练数据 文件名[0-9]+.json 数字按顺序依次递增
        val   验证数据 文件名[0-9]+.json 数字按顺序依次递增
        test  测试数据 文件名[0-9]+.json 数字按顺序依次递增
    """
    def __init__(self, split: str, datetype: str='cnndm', path: str=None) -> None:
        assert split in ['train', 'val', 'test']
        assert datetype in ['bytecup2018', 'cnndm']

        if path is None:
            sysstr = platform.system()
            if sysstr == 'Linux':
                self.path = r'/home/webdev/ai/competition/{}/data'.format(datetype)

            elif sysstr == "Darwin":
                self.path = r'/Users/oneai/ai/data/{}'.format(datetype)
        else:
            self.path = path # 源路径
        self._data_path = join(self.path, split) # 数据路径
        self._n_data = self.count_json(self._data_path)

        self.FILE_PRE = '{}.json'
        self.split = split

    def __len__(self) -> int:
        return self._n_data

    # def __getitem__(self, i: int):
    #     filename = join(self._data_path, self.FILE_PRE.format(self.split, i))
    #     df = pd.read_json(filename, lines=True)
    #     return zip(df['content'].values, df['title'].values)

    def __getitem__(self, i: int):
        with open(join(self._data_path, '{}.json'.format(i))) as f:
            js = json.loads(f.read())
        return js

    def count_json(self, path):
        """
            计算json文件个数
            count number of data in the given path
        """
        matcher = re.compile(r'[0-9]+\.json')
        match = lambda name: bool(matcher.match(name))
        names = os.listdir(path)
        n_data = len(list(filter(match, names)))
        return n_data

    def count_txt(self, path):
        """
            计算txt文件个数
            count number of data in the given path
        """
        matcher = re.compile(r'.*[0-9]+\.txt')
        match = lambda name: bool(matcher.match(name))
        names = os.listdir(path)
        n_data = len(list(filter(match, names)))
        return n_data



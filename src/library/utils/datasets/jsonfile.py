""" CNN/DM dataset"""
import os
import re
import json
from os.path import join
from torch.utils.data import Dataset


class JsonFileDataset(Dataset):
    """
        trainer 训练数据 文件名[0-9]+.json 数字按顺序依次递增
        val   验证数据 文件名[0-9]+.json 数字按顺序依次递增
        test  测试数据 文件名[0-9]+.json 数字按顺序依次递增
    """
    def __init__(self, split: str, path: str=None) -> None:
        assert split in ['train', 'val', 'test']

        self.path = path # 源路径
        self._data_path = join(self.path, split) # 数据路径
        self._n_data = self.count_json(self._data_path)

        self.FILE_PRE = '{}.json'
        self.split = split

    def __len__(self) -> int: # 必须相应，否则批处理器，不知道加载多少
        return self._n_data

    def __getitem__(self, i: int):
        with open(join(self._data_path, '{}.json'.format(i))) as f:
            js = json.loads(f.read())
        return js

    def count_json(self, path):
        """
            追加的统计文件个数的方法，计算json文件个数
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


def count_train_txt(path):
    """
        计算txt文件个数
        count number of data in the given path
    """
    matcher = re.compile(r'.*trainer.[0-9]+\.txt')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data


def count_txt(path):
    """
        计算txt文件个数
        count number of data in the given path
    :param path:
    :return:
    """
    matcher = re.compile(r'[0-9]+\.txt')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data


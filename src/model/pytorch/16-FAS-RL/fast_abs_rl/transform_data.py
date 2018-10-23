import sys
from data.data import convert_p2j


if __name__ == "__main__":

    if sys.platform.startswith('linux'):
        dst_path = "/home/webdev/ai/competition/bytecup2018/data/train/"
        raw_path = "/home/webdev/ai/competition/bytecup2018/data/raw"
        voc_path = "/home/webdev/ai/competition/bytecup2018/data/"
    elif sys.platform.startswith('darwin'):
        dst_path = "/Users/oneai/ai/data/bytecup/train/"
        raw_path = "/Users/oneai/ai/data/bytecup/raw"
        voc_path = "/Users/oneai/ai/data/bytecup"

    convert_p2j(raw_path, dst_path, voc_path)
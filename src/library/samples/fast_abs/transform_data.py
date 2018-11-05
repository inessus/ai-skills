import sys
sys.path.append("../../..")

from library.utils.transforms.transforms import convert_p2j, convert_txt_json


if __name__ == "__main__":

    if sys.platform.startswith('linux'):
        dst_path = "/media/webdev/store/competition/bytecup2018/data/trainer/"
        raw_path = "/media/webdev/store/competition/bytecup2018/data/raw"
        voc_path = "/media/webdev/store/competition/bytecup2018/data/"
        val_path = "/media/webdev/store/competition/bytecup2018/data/val"
    elif sys.platform.startswith('darwin'):
        dst_path = "/Users/oneai/ai/data/bytecup/trainer/"
        raw_path = "/Users/oneai/ai/data/bytecup/raw"
        voc_path = "/Users/oneai/ai/data/bytecup"

    convert_p2j(raw_path, dst_path, voc_path)
    convert_txt_json(val_path, val_path)

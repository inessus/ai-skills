# 对于图像，常用的有：Pillow，OpenCV
# 对于语音，常用的有：scipy, libosa
# 对于文本，常用的有：NLTK, SpaCy


# torch.nn包仅支持对批量数据的处理,而不能对单个样本进行处理.
# 例如,nn.Conv2d只接受4维的张量:
# nSamples * nChannels * Height * Width
# 如果只有单个样本,那么使用input.unsqueeze(0)来增加假的batch维度.
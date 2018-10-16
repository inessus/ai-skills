import jieba.analyse
import jieba
# https://pan.baidu.com/s/1ggA4QwN

from gensim.models import word2vec
import logging

jieba.suggest_freq('沙瑞金',True) # 可调节单个词语的词频，使其能（或不能）被分出来
jieba.suggest_freq('田国富', True)
jieba.suggest_freq('高育良', True)
jieba.suggest_freq('侯亮平', True)
jieba.suggest_freq('钟小艾', True)
jieba.suggest_freq('陈岩石', True)
jieba.suggest_freq('欧阳菁', True)
jieba.suggest_freq('易学习', True)
jieba.suggest_freq('王大路', True)
jieba.suggest_freq('蔡成功', True)
jieba.suggest_freq('孙连城', True)
jieba.suggest_freq('季昌明', True)
jieba.suggest_freq('丁义珍', True)
jieba.suggest_freq('郑西坡', True)
jieba.suggest_freq('赵东来', True)
jieba.suggest_freq('高小琴', True)
jieba.suggest_freq('赵瑞龙', True)
jieba.suggest_freq('林华华', True)
jieba.suggest_freq('陆亦可', True)
jieba.suggest_freq('刘新建', True)
jieba.suggest_freq('刘庆祝', True)
jieba.suggest_freq('京州市', True)
jieba.suggest_freq('副市长', True)


with open('./src/model/nlp/in_the_name_of_people.txt','rb') as f:
    document  = f.read()
    document_cut = jieba.cut(document, cut_all =False)
   # print('/'.join(document_cut))
    result = ' '.join(document_cut)
    result = result.encode('utf-8')
    
    with open('/Users/oneai/ai/NLP/data/in_the_name_of_people_segment.txt','wb+') as f1:
         f1.write(result)#读取的方式和写入的方式要一致
f.close()
f1.close()


sentences = word2vec.Text8Corpus('/Users/oneai/ai/NLP/data/in_the_name_of_people_segment.txt')
model = word2vec.Word2Vec(sentences, size = 100, hs=1, min_count=1, window=3)
#模型的预测

print('-----------------分割线---------------------------')
#保留模型，方便重用
model.save(u'人民的名义.model')

print('-----------------分割线----------------------------')
#计算两个词向量的相似度
try:
    sim1 = model.similarity(u'沙瑞金',u'高育良')
    sim2 = model.similarity(u'李达康',u'易学习')
except KeyError:
    sim1 = 0
    sim2 = 0
print(u'沙瑞金 和 高育良 的相似度为 ',sim1)
print(u'李达康 和 易学习 的相似度为 ',sim2)

print('-----------------分割线---------------------------')
#与某个词（李达康）最相近的3个字的词
print(u'与李达康最相近的3个字的词')
req_count = 5
for key in model.similar_by_word(u'李达康',topn =100):
    if len(key[0])==3:
        req_count -=1
        print(key[0],key[1])
        if req_count ==0:
            break
            
print('-----------------分割线---------------------------')
#计算某个词(侯亮平)的相关列表
try:
    sim3 = model.most_similar(u'侯亮平',topn =20)
    print(u'和 侯亮平 与相关的词有：\n')
    for key in sim3:
        print(key[0],key[1])
except:
    print(' error')
    
print('-----------------分割线---------------------------')
#找出不同类的词
sim4 = model.doesnt_match(u'沙瑞金 高育良 李达康 刘庆祝'.split())
print(u'这类人物中不同类的人名',sim4)
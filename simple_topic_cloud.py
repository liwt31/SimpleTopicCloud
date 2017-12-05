import argparse
import time
import gensim
import jieba
import jieba.analyse
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt


def read2unicode(fname):
    with open(fname, 'rb') as fin:
        raw_text = fin.read()
    try:
        return raw_text.decode('utf-8')
    except UnicodeDecodeError:
        pass
    return raw_text.decode('gbk')


def build_model(corpus_fname):
    corpus_list = read2unicode(corpus_fname).splitlines()
    print('开始分词...')
    time1 = time.time()
    lines = [jieba.lcut(corpus) for corpus in corpus_list]
    print('分词时间 %f s' % (time.time() - time1))
    print('开始训练Word2Vec模型')
    time1 = time.time()
    model = gensim.models.Word2Vec(lines)
    print('word2vec模型训练时间 %f s' % (time.time() - time1))
    return model


def analyze_target(target_fname, model):
    target_text = read2unicode(target_fname)
    print('开始提取目标文本关键词')
    time1 = time.time()
    kw_list = jieba.analyse.extract_tags(target_text, topK=500, withWeight=True,
                                         allowPOS=['n', 'v', 'nr', 'ns', 'vn', 'a', 'l'])
    print('提取关键词时间 %f s' % (time.time() - time1))
    kw_weight = pd.Series({k: v for k, v in kw_list}, name='weight')
    kw_vector = pd.DataFrame({k: model.wv[k] for k, v in kw_list if k in model.wv}).transpose()
    n_kw_vector = kw_vector.div(kw_vector.std(axis=1), axis=0)
    filtered_kw_weight = kw_weight[n_kw_vector.index]
    ac = AgglomerativeClustering(30)
    ac.fit(kw_vector)
    kw_label = pd.Series(ac.labels_, index=kw_vector.index, name='label')
    tsne = TSNE()
    print('开始进行t-SNE降维')
    time1 = time.time()
    kw_tsne_v = tsne.fit_transform(n_kw_vector)
    print('t-SNE降维时间: %f s' % (time.time() - time1))
    kw_tsne_df = pd.DataFrame(kw_tsne_v, index=n_kw_vector.index, columns=['x', 'y'])
    kw_df = pd.concat([kw_label, kw_tsne_df, filtered_kw_weight], axis=1)
    return kw_df

def draw_fig(kw_df, output_fname):
    print('开始绘图....')
    plt.figure(figsize=(18, 18))
    axis = plt.subplot(111)
    plt.axis('off')
    axis.scatter(kw_df['x'], kw_df['y'], s=np.sqrt(kw_df['weight']) * 3000, alpha=0.6, c=kw_df['label'], cmap='jet')
    for index, row in kw_df.iterrows():
        axis.annotate('%d%s' % (row['label'], index), (row['x'], row['y']), alpha=0.8)
    if not output_fname:
        output_fname = '%s.png' % time.strftime('%y%m%d-%H%M%S')
    if '.' not in output_fname:
        output_fname += '.png'
    plt.savefig(output_fname, dpi=200)
    print('图片已保存至%s' % output_fname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='为目标文本生成主题词云图片')
    parser.add_argument('target_fname',
                      help='欲提取主题制作词云的目标文本文件名')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-c', dest='corpus_fname',
                      help='训练Word2Vec模型语料库文件名，文件中每行为一个句子或一段话')
    group.add_argument('-l', dest='load_model_fname',
                      help='读取Word2Vec模型文件名')
    parser.add_argument('-s', dest='save_model_fname',
                      help='存储Word2Vec模型文件名')
    parser.add_argument('-o', dest='output_fname',
                      help ='输出图片文件名，默认为以时间为文件名的png格式图片')
    args = parser.parse_args()
    target_fname = args.target_fname
    corpus_fname = args.corpus_fname
    load_model_fname = args.load_model_fname
    save_model_fname = args.save_model_fname
    output_fname = args.output_fname
    if corpus_fname:
        print('从%s训练Word2Vec模型' % corpus_fname)
        model = build_model(corpus_fname)
    elif load_model_fname:
        print('从%s读取Word2Vec模型' % load_model_fname)
        model = gensim.models.word2vec.Word2Vec.load(load_model_fname)
    else:
        print('Should not be here')
        exit()
        model = None
    if model and save_model_fname:
        model.save(save_model_fname)
    kw_df = analyze_target(target_fname, model)
    draw_fig(kw_df, output_fname)
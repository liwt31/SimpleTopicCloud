# 简单词云生成 Simple Topic Cloud Generator
* 利用Word2Vec对关键词（主题词）进行表示， AgglomerativeClustering进行聚类，t-SNE进行降维可视化。绘图采用`matplotlib`。
* 示例图片（相对位置表示词的相对相似度， 大小表示词的权重，颜色及词的前缀表示词的类别）：
![image](https://github.com/liwt31/SimpleTopicCloud/raw/master/china_weibo.png)

## 环境要求（全部可以通过`pip`安装）
* Python3
* [Gensim](https://github.com/RaRe-Technologies/gensim)，用于Word2Vec模型
* [sklearn](http://scikit-learn.org/stable/)，用于聚类及降维
* [Pandas](http://pandas.pydata.org/), numpy
* [matplotlib](http://matplotlib.org/)，用于绘图
* [结巴分词](https://github.com/fxsjy/jieba)，用于分词

## 如何使用
* 在TutorialNotebook文件夹内，保存了用于演示的jupyter-notebook版本，可以生成本说明中的示例图片
* 或者直接使用simple_topic_cloud为目标文本绘制词云，可以通过`python simple_topic_cloud.py -h`查看帮助信息：
```
usage: simple_topic_cloud.py [-h] [-c CORPUS_FNAME | -l LOAD_MODEL_FNAME]
                             [-s SAVE_MODEL_FNAME] [-o OUTPUT_FNAME]
                             target_fname

为目标文本生成主题词云图片

positional arguments:
  target_fname         欲提取主题制作词云的目标文本文件名

optional arguments:
  -h, --help           show this help message and exit
  -c CORPUS_FNAME      训练Word2Vec模型语料库文件名，文件中每行为一个句子或一段话
  -l LOAD_MODEL_FNAME  读取Word2Vec模型文件名
  -s SAVE_MODEL_FNAME  存储Word2Vec模型文件名
  -o OUTPUT_FNAME      输出图片文件名，默认为以时间为文件名的png格式图片

```

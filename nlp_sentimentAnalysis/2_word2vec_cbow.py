
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import pickle
import string
import requests
import collections
import io
import tarfile
import urllib.request
import text_helpers
#from nltk.corpus import stopwords
from tensorflow.python.framework import ops

#显示中文
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
#%matplotlib inline


ops.reset_default_graph()
#改变当前工作目录到指定的路径
#os.chdir("E:/yanyan/")

#数据样本目录
data_folder_name = 'training_data'
if not os.path.exists(data_folder_name):
    os.makedirs(data_folder_name)

#新建回话
sess = tf.Session()

# 声明模型超参数
batch_size = 500
#生成词向量大小
embedding_size = 200
#词汇量大小
vocabulary_size = 2000
#迭代次数
generations = 50000
model_learning_rate = 0.001
#负采样样本量
num_sampled = int(batch_size/2)
#窗口大小
window_size = 3


print_valid_every = 5000
print_loss_every = 100

# 停用词
stops=text_helpers.stopwords(data_folder_name)
#stops = stopwords.words('english')

#用于测试的词
valid_words = ['love', 'hate', 'happy', 'sad', 'man', 'woman']
#加载数据
texts, target = text_helpers.load_movie_data(data_folder_name)
# 标准化数据   去掉标点符号 数字  停用词 去掉多余的空格
texts = text_helpers.normalize_text(texts, stops)

# 去掉较短的语句，至少要包含3个单词
#target = [target[ix] for ix, x in enumerate(texts) if len(x.split()) > 2]
texts = [x for x in texts if len(x.split()) > 2]
    
# 构建单词字典,给每一个单词编号*****
word_dictionary = text_helpers.build_dictionary(texts, vocabulary_size)


#反过来构建单词字典，用每一编号表示每一个单词
word_dictionary_rev = dict(zip(word_dictionary.values(), word_dictionary.keys()))
#将文本数据中的单词转化为数字表达
text_data = text_helpers.text_to_numbers(texts, word_dictionary)
#将测试单词用数字表达
valid_examples = [word_dictionary[x] for x in valid_words]    

# 嵌入层
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

# NCE 输出层参数
nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],stddev=1.0 / np.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

# 声明占位符
x_inputs = tf.placeholder(tf.int32, shape=[batch_size, 2*window_size])
y_target = tf.placeholder(tf.int32, shape=[batch_size, 1])

valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

# 根据索引查找每个词的向量
# 将窗口内的每个词的词向量相加-->成一个向量
embed = tf.zeros([batch_size, embedding_size])
for element in range(2*window_size):
    embed += tf.nn.embedding_lookup(embeddings, x_inputs[:, element])

# nce 损失函数 有个负采样过程  注意参数的顺序
loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, y_target,embed,num_sampled, vocabulary_size))
                                     
# 创建优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate=model_learning_rate).minimize(loss)

# 词的相似度--余弦相似度
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)


#Add variable initializer.
init = tf.initialize_all_variables()
sess.run(init)


loss_vec = []
loss_x_vec = []

#每次迭代的数据的数据结构
text_helpers.generate_batch_data(text_data, batch_size,window_size, method='cbow')
#训练模型
for i in range(generations):
    batch_inputs, batch_labels = text_helpers.generate_batch_data(text_data, batch_size,window_size, method='cbow')
    feed_dict = {x_inputs : batch_inputs, y_target : batch_labels}
    # 每次迭代
    sess.run(optimizer, feed_dict=feed_dict)
    # 计算每100次迭代的损失
    if (i+1) % print_loss_every == 0:
        loss_val = sess.run(loss, feed_dict=feed_dict)
        loss_vec.append(loss_val)
        loss_x_vec.append(i+1)
        print('在第 {}次的 损失: {}'.format(i+1, loss_val))
    # 计算每5000迭代和测试词相似度最近的5个词
    if (i+1) % print_valid_every == 0:
        sim = sess.run(similarity, feed_dict=feed_dict)
        for j in range(len(valid_words)):
            valid_word = word_dictionary_rev[valid_examples[j]]
            top_k = 5 # number of nearest neighbors
            nearest = (-sim[j, :]).argsort()[1:top_k+1]
            log_str = "Nearest to {}:".format(valid_word)
            for k in range(top_k):
                close_word = word_dictionary_rev[nearest[k]]
                log_str = '{} {},' .format(log_str, close_word)
            print(log_str)

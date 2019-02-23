import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from os import listdir
from os.path import isfile, join
import re
from random import randint
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
#%matplotlib inline


#加载次列表
wordsList = np.load('./training_data/wordsList.npy')
wordsList = wordsList.tolist()
#编码
wordsList = [word.decode('UTF-8') for word in wordsList]
#词向量
wordVectors = np.load('./training_data/wordVectors.npy')
#查看词向量的维度
print(len(wordsList))
print(wordVectors.shape)
#查找某个单词向量
baseballIndex = wordsList.index('baseball')
wordVectors[baseballIndex]
#构造一个句子的向量组例如影评数据：I thought the movie was incredible and inspiring 8个单词
#句子的最大长度
maxSeqLength = 10 
#每个词向量的维度
numDimensions = 300
firstSentence = np.zeros((maxSeqLength), dtype='int32')
firstSentence[0] = wordsList.index("i")
firstSentence[1] = wordsList.index("thought")
firstSentence[2] = wordsList.index("the")
firstSentence[3] = wordsList.index("movie")
firstSentence[4] = wordsList.index("was")
firstSentence[5] = wordsList.index("incredible")
firstSentence[6] = wordsList.index("and")
firstSentence[7] = wordsList.index("inspiring")
#仔细观察最后两个单词为0 ,长度不够用0填补,长度太长截断
print(firstSentence.shape)
print(firstSentence)
#得到每个单词的向量
print(wordVectors[firstSentence[0]])
print(wordVectors[firstSentence[1]])
print(wordVectors[firstSentence[2]])

#得到一句话中每个词的向量上面的步骤过于复杂,可以直接使用tf.nn.embedding_lookup函数例如：
with tf.Session() as sess:
    print(tf.nn.embedding_lookup(wordVectors,firstSentence).eval().shape)


#统计评论数据
numWords = []
#正面评论数据
with open("training_data/positiveReviews.txt", "r", encoding='utf-8') as f:
    for line in f.readlines():
        counter = len(line.split())
        numWords.append(counter)
#负面评论数据
with open("training_data/negativeReviews.txt", "r", encoding='utf-8') as f:
    for line in f.readlines():
        counter = len(line.split())
        numWords.append(counter)
numFiles = len(numWords)
print('评论总条数', numFiles)
print('单词总数', sum(numWords))
print('平均每条评论的单词数', sum(numWords)/len(numWords))

#可视化评论数据
import matplotlib.pyplot as plt
#%matplotlib inline
plt.hist(numWords, 50)
plt.xlabel('长度')
plt.ylabel('频率')
plt.axis([0, 1200, 0, 8000])
plt.show()

#从直方图和句子的平均单词数，我们认为将句子最大长度设置为 250 是可行的
maxSeqLength = 250
#举一个例子将第一句评论转化为索引矩阵
#数据清洗
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

firstFile = np.zeros((maxSeqLength), dtype='int32')
with open("training_data/positiveReviews.txt", "r", encoding='utf-8') as f:
    indexCounter = 0
    line=f.readline()
    cleanedLine = cleanSentences(line.strip())
    split = cleanedLine.split()
    for word in split:
        try:
            firstFile[indexCounter] = wordsList.index(word)
        except ValueError:
            firstFile[indexCounter] = 399999 #Vector for unknown words
        indexCounter = indexCounter + 1
print(firstFile)


#定义并计算整个评论的索引矩阵，过程太慢不演示
"""ids = np.zeros((numFiles, maxSeqLength), dtype='int32')
fileCounter = 0
with open("training_data/positiveReviews.txt", "r", encoding='utf-8') as f:
    for line in f.readlines():
        indexCounter=0
        cleanedLine = cleanSentences(line.strip())
        split = cleanedLine.split()
        for word in split:
            try:
                ids[fileCounter][indexCounter] = wordsList.index(word)
            except ValueError:
                ids[fileCounter][indexCounter] = 399999 #Vector for unkown words
            indexCounter = indexCounter + 1
            if indexCounter >= maxSeqLength:
                break
        fileCounter = fileCounter + 1 

with open("training_data/negativeReviews.txt", "r", encoding='utf-8') as f:
    for line in f.readlines():
        indexCounter=0
        cleanedLine = cleanSentences(line.strip())
        split = cleanedLine.split()
        for word in split:
            try:
                ids[fileCounter][indexCounter] = wordsList.index(word)
            except ValueError:
                ids[fileCounter][indexCounter] = 399999 #单词表中查不到该单词
            indexCounter = indexCounter + 1
            if indexCounter >= maxSeqLength:
                break
        fileCounter = fileCounter + 1
np.save('idsMatrix_1', ids)
"""
#直接load
ids = np.load('./training_data/idsMatrix.npy')

print("索引矩阵 载入完")

#辅助函数
def getTrainBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        if (i % 2 == 0): 
            num = randint(1,11499)
            labels.append([1,0])
        else:
            num = randint(13499,24999)
            labels.append([0,1])
        arr[i] = ids[num-1:num]
    return arr, labels
def getTestBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        num = randint(11499,13499)
        if (num <= 12499):
            labels.append([1,0])
        else:
            labels.append([0,1])
        arr[i] = ids[num-1:num]
    return arr, labels

#超参数
batchSize = 24
units = 64
numClasses = 2
iterations = 50000
#iterations = 2500



tf.reset_default_graph()
#输入输出占位符
labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

#调用 tf.nn.embedding_lookup() 函数来得到我们的词向量。该函数最后将返回一个三维向量，第一个维度是批处理大小，第二个维度是句子长度，第三个维度是词向量长度
data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors,input_data)

#初始化RNN单元的类型
#cell = tf.contrib.rnn.BasicRNNCell(units)
cell = tf.contrib.rnn.BasicLSTMCell(units)
cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=0.75)
#dynamic RNN 函数的第一个输出可以被认为是最后的隐藏状态向量。这个向量将被重新确定维度，然后乘以最后的权重矩阵和一个偏置项来获得最终的输出值
#展开网络
value, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
#最后一层输出的权重矩阵
weight = tf.Variable(tf.truncated_normal([units, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
value = tf.transpose(value, [1, 0, 2])
#得到展开网络后的最后一层,将RNN或者LSTM单元的的的最后一次的输出做为整个单元的输出
last = tf.gather(value, int(value.get_shape()[0]) - 1)
#网络的输出
prediction = (tf.matmul(last, weight) + bias)

#定义正确的预测函数和正确率评估参数。正确的预测形式是查看最后输出的0-1向量是否和标记的0-1向量相同。
correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
#标准的交叉熵损失函数来作为损失值。对于优化器，我们选择 Adam，并且采用默认的学习率。
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)

#tf.InteractiveSession()来构建会话的时候，我们可以先构建一个session然后再定义操作(operation)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
#训练网络
loss_s = []
accuracy_s=[]
for i in range(iterations):
    #获取每次迭代的数据
    nextBatch, nextBatchLabels = getTrainBatch();
    res=sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels}) 
    if (i % 1000 == 0 and i != 0):
        loss_ = sess.run(loss, {input_data: nextBatch, labels: nextBatchLabels})
        accuracy_ = sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})
        print("迭代次数 {}/{}...".format(i+1, iterations),"损失 {}...".format(loss_),"准确率 {}...".format(accuracy_))
        loss_s.append(loss_)
        accuracy_s.append(accuracy_)
    if (i % 10000== 0 and i != 0):
        save_path = saver.save(sess, "models/lstm/pretrained.ckpt", global_step=i)
        


#sess = tf.InteractiveSession()
#saver = tf.train.Saver()
#saver.restore(sess, tf.train.latest_checkpoint('models'))
#测试集
iterations = 10
for i in range(iterations):
    nextBatch, nextBatchLabels = getTestBatch();
    print("准确率:", (sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})) * 100)
#每次迭代的损失函数
x= np.arange(1000,iterations,1000)
plt.plot(x,loss_s, 'k-', label='损失')
plt.plot(x,accuracy_s, 'r--', label='准确率')
plt.title('LSTM模型')
plt.xlabel('迭代次数')
plt.legend(loc='upper right')
plt.show()
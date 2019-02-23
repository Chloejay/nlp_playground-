
import string
import os
import urllib.request
import io
import tarfile
import collections
import numpy as np


#停用词
def stopwords(training_data):
    sp=[]
    f=open("training_data/stopwords.txt","r",encoding='utf8')
    for text in f.readlines():
        sp.append(text.strip())
    f.close()
    return sp


# 标准化
def normalize_text(texts, stops):
    # 转化为小写
    texts = [x.lower() for x in texts]
    # 去掉标点符号
    texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]
    # 去掉数字
    texts = [''.join(c for c in x if c not in '0123456789') for x in texts]
    # 去掉停用词
    texts = [' '.join([word for word in x.split() if word not in (stops)]) for x in texts]
    # 去掉多余的空格
    texts = [' '.join(x.split()) for x in texts]
    return(texts)


# 构建单词字典
def build_dictionary(sentences, vocabulary_size):
    # 将一行句子转化为 单词 list
    split_sentences = [s.split() for s in sentences]
    words = [x for sublist in split_sentences for x in sublist]
    #初始化一个列表[word, word_count] ,未知的单词用-1标识
    count = [['RARE', -1]]
    #讲单词按照词频从高到低取（vocabulary_size-1）个单词
    count.extend(collections.Counter(words).most_common(vocabulary_size-1))
    #构建单词字典
    word_dict = {}
    #每一单词用一个数字表示
    for word, word_count in count:
        word_dict[word] = len(word_dict)
    return(word_dict)


# 将文本数据中的单词转化为数字表达
def text_to_numbers(sentences, word_dict):
    # 初始化一个数据列表
    data = []
    for sentence in sentences:
        sentence_data = []
        # 用构建好的单词字典去查找每个单词对应的数字
        for word in sentence.split():
            if word in word_dict:
                word_ix = word_dict[word]
            else:
                word_ix = 0
            sentence_data.append(word_ix)
        data.append(sentence_data)
    return(data)
    

# Generate data randomly (N words behind, target, N words ahead)
def generate_batch_data(sentences, batch_size, window_size, method='skip_gram'):
    # Fill up data batch
    batch_data = []
    label_data = []
    while len(batch_data) < batch_size:
        # select random sentence to start
        rand_sentence_ix = int(np.random.choice(len(sentences), size=1))
        
        rand_sentence = sentences[rand_sentence_ix]
        # Generate consecutive windows to look at
        window_sequences = [rand_sentence[max((ix-window_size),0):(ix+window_size+1)] for ix, x in enumerate(rand_sentence)]
        # Denote which element of each window is the center word of interest
        label_indices = [ix if ix<window_size else window_size for ix,x in enumerate(window_sequences)]
        
        # Pull out center word of interest for each window and create a tuple for each window
        if method=='skip_gram':
            batch_and_labels = [(x[y], x[:y] + x[(y+1):]) for x,y in zip(window_sequences, label_indices)]
            # Make it in to a big list of tuples (target word, surrounding word)
            tuple_data = [(x, y_) for x,y in batch_and_labels for y_ in y]
            batch, labels = [list(x) for x in zip(*tuple_data)]
        elif method=='cbow':
            
            batch_and_labels = [(x[:y] + x[(y+1):], x[y]) for x,y in zip(window_sequences, label_indices)]
            # Only keep windows with consistent 2*window_size
            batch_and_labels = [(x,y) for x,y in batch_and_labels if len(x)==2*window_size]
            #print(batch_and_labels)
            if(len(batch_and_labels)==0):
                continue
            batch, labels = [list(x) for x in zip(*batch_and_labels)]
            
            
        elif method=='doc2vec':
            # For doc2vec we keep LHS window only to predict target word
            batch_and_labels = [(rand_sentence[i:i+window_size], rand_sentence[i+window_size]) for i in range(0, len(rand_sentence)-window_size)]
            batch, labels = [list(x) for x in zip(*batch_and_labels)]
            # Add document index to batch!! Remember that we must extract the last index in batch for the doc-index
            batch = [x + [rand_sentence_ix] for x in batch]
        else:
            raise ValueError('Method {} not implmented yet.'.format(method))
            
        # extract batch and labels
        batch_data.extend(batch[:batch_size])
        label_data.extend(labels[:batch_size])
    # Trim batch and label at the end
    batch_data = batch_data[:batch_size]
    label_data = label_data[:batch_size]
    
    # Convert to numpy array
    batch_data = np.array(batch_data)
    label_data = np.transpose(np.array([label_data]))
    
    return(batch_data, label_data)
    
    
#加载影评数据
def load_movie_data(data_folder_name):
    pos_file = os.path.join(data_folder_name, 'rt-polarity.pos')
    neg_file = os.path.join(data_folder_name, 'rt-polarity.neg')
    pos_data = []
    with open(pos_file, 'r',encoding='utf8') as temp_pos_file:
        for row in temp_pos_file:
            pos_data.append(row)
    neg_data = []
    with open(neg_file, 'r',encoding='utf8') as temp_neg_file:
        for row in temp_neg_file:
            neg_data.append(row)
    texts = pos_data + neg_data
    target = [1]*len(pos_data) + [0]*len(neg_data)
    return(texts, target)
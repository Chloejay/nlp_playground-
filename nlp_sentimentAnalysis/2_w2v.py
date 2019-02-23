import gensim.models.word2vec as w2v
sentences = w2v.LineSentence('training_data/yitiantulong.txt')  
model = w2v.Word2Vec(sentences, size=100, window=3, min_count=5, workers=4,iter=10) 


print(list(model['张无忌']))
print(model.similar_by_word('张无忌'))
#print(model.most_similar('张无忌', topn=20))
#print(model.score("张无忌 赵敏".split(" ")))
#print(model.most_similar(['张无忌', '赵敏'], ['周芷若'], topn=3))
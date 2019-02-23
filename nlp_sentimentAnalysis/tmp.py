from nltk.corpus import stopwords
stops = stopwords.words('english')

f1=open("training_data/stopwords.txt","w",encoding='utf8')
for stop in stops:
    f1.write(stop+"\n")
f1.flush()
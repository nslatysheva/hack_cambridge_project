import numpy as np
import scipy as sp
import re
import sys
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA

assert len(sys.argv) == 2
f = open('./'+sys.argv[1])
authors=f.read()
f.close()
authors=authors.split()

corpus = []
minibatch_corpus=[]
minibatch_size=10

g = open('./sentence_batch_per_author.csv', 'w')

for author in authors:
	e = open('./'+author+'_sentences.txt', 'w')
	print author
	filename='./'+author+'/'+author+'.txt'
	f = open(filename)
	text = f.read()
	f.close()
	text=text.decode(encoding='utf-8', errors='ignore').encode(encoding='ascii', errors='ignore')
	text=re.sub(r'\.\s', r'\n', text)
	text=re.sub(r"[^A-Za-z0-9'\n]", r' ', text)
	text=re.sub(r"\s\'\s", r' ', text)
	text = re.sub(r' +', r' ', text)
	text = re.sub(r'\n+', r'\n', text)
	text = text.upper().splitlines()
	text_minibatch=[]
	for sentence in text:
		sentence = ' '.join(sentence.split())
		if sentence == '':
			continue
		sentence=re.sub(r"(\s'\s)+", r' ', sentence)
		e.write(sentence+'\n')
		text_minibatch.append(sentence)
	author_minibatch_corpus=[]
	for i in xrange(int(len(text_minibatch)/minibatch_size)):
		text_minibatch_slice = text_minibatch[i*minibatch_size:(i+1)*minibatch_size]
		joined_txt=' '.join(text_minibatch_slice)
		string = joined_txt+' , '+author+'\n'
		g.write(string)
		author_minibatch_corpus.append(joined_txt)
	minibatch_corpus.append(author_minibatch_corpus)
	text = ' '.join(text_minibatch)
	corpus.append(text)
	e.close()
g.close()

f = open('test_corpus.txt', 'w')
for line in corpus:
	f.write(line+'\n')
f.close()

for author, sentences in zip(authors, minibatch_corpus):
	f = open(author+'_minibatch.txt', 'w')
	for sentence in sentences:
		f.write(sentence+'\n')
	f.close()

#TFIDF=TfidfVectorizer(input='content', strip_accents='ascii', stop_words='english', lowercase=True, analyzer='word', ngram_range=(1,1), min_df = 0)
#tfidf = TFIDF.fit_transform(corpus)
#tfidf_minibatch = [TFIDF.transform(author_minibatch) for author_minibatch in minibatch_corpus]





from collections import defaultdict
import re
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim import corpora, models, similarities, matutils
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
from gensim.matutils import cossim
from nltk.corpus import stopwords
from nltk import word_tokenize, sent_tokenize
import pickle
import random
import sys
from tqdm import tqdm
from sklearn import manifold

# from nltk import download
# download()

project_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) + '/'
data_path = project_path + 'data/'
models_path = project_path + 'models/'
quijote_file = 'el_quijote.txt'
with open(data_path+quijote_file, "r", encoding='utf-8') as f:
    corpus = f.read() # read corpus

############################ ir corriendo ##########
# remplazo los signos $
corpus = corpus.replace("$"," signopesos ")
# reemplazo numeros con " num " y  paso a minuscula
reg_num = re.compile(r"\d+[.,]?\d*") # Regular expression to search numbers
corpus = reg_num.sub(" NUM ",corpus).lower()
corpus = sent_tokenize(corpus)
# tiro los tokens no alphabeticos
trainset = []
for sent in tqdm(corpus):
    tokens = []#corpus
    for token in word_tokenize(sent):
        if token.isalpha():
            tokens.append(token)
    trainset.append(tokens)

print ("el corpus tiene",len(trainset), "oraciones y",sum([len(x) for x in trainset]),"palabras"   )

# filtro oraciones cortas
trainset2 = []
for sent in trainset:
    if len(sent)>3:
        trainset2.append(sent)

print( "el corpus tiene",len(trainset2), "oraciones y",sum([len(x) for x in trainset2]),"palabras"  )


# "window" es el tamaño de la ventana. windows = 10, usa 10 palabras a la izquierda y 10 palabras a la derecha
# "n_dim" es la dimension (i.e. el largo) de los vectores de word2vec
# "workers" es el numero de cores que usa en paralelo. Para aprobechar eso es necesario tener instalado Cython)
# "sample": word2vec filtra palabras que aparecen una fraccion mayor que "sample"
# "min_count": Word2vec filtra palabras con menos apariciones que  "min_count"
# "sg": para correr el Skipgram model (sg = 1), para correr el CBOW (sg = 0)
# para mas detalle ver: https://radimrehurek.com/gensim/models/word2vec.html
n_dim = 20
w2v_model = Word2Vec(trainset2, workers=3,size=n_dim, min_count = 10, window = 10, sample = 1e-3,negative=10,sg=1)


w2v_model.save(models_path + "word2vec_quijote")  # save model
w2v_model = Word2Vec.load(models_path + "word2vec_quijote")  # load model

print ("quijote-locura similarity:",w2v_model.wv.n_similarity(["quijote"], ["locura"]))
w2v_model.most_similar(positive=["quijote"], negative=[], topn=25)


topn = 20
len_answer = 20
starting_sentence = "Vaya a comer con los molinos"
starting_sentence_list = starting_sentence.lower().split()
answer_list = []
for i in range(len_answer):
    words = [''] * topn
    proba = [0] * topn
    for i, (w, p) in enumerate(w2v_model.predict_output_word(starting_sentence_list + answer_list, topn=topn)):
        if w not in answer_list:
            words[i] = w
            proba[i] = p
    # next_word = np.random.choice(words, p=np.array(proba)/np.array(proba).sum())
    next_word = words[int(np.argmax(proba))]
    answer_list.append(next_word)
answer = ' '.join(answer_list)
print(starting_sentence)
print(answer)





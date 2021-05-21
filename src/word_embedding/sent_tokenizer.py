import numpy as np
from tqdm import tqdm
import re
from konlpy.tag import Mecab
import gensim
from gensim.models import FastText

### Using FastText(skip-gram) Model for Word2Vec
'''
Architecture : 
FastText(sentences= train_ft,\
        size= 768,\
        window= 5,\
        min_count= 1,\
        workers= 7,\
        sg = 1,\
        min_n= 1,\
        max_n=6,\
        iter=10)
'''

# load finetuned word2vec model
fname = '/repo/course/sem21_01/youtube_summarizer/src/word_embedding/model/fasttext.model'
ft_model = FastText.load(fname)

# define mecab tokenizer
mecab = Mecab()

# define stopwords
stopwords = ['년','월','일','의','은','는','이','가','좀','잘','과','도','을','를','으로','자','에','와','한','하다','합니다','이다','입니다','습니다'] # update stopwords

# define sentence tokenizer
def get_sent_token(sentence, stopwords = stopwords):
    '''
    tokenize input sentence when given using stopwords
    '''
    sent_token = re.sub('[^가-힣a-z]', ' ', sentence) # 영어 소문자와 한글을 제외한 모든 문자를 제거
    sent_token = mecab.morphs(sent_token) # 토큰화
    sent_token = [word for word in sent_token if not word in stopwords] # 불용어 제거

    return sent_token

def get_sent_embedding(sent_token):
    '''
    get sentence token and returns sentece embedding value
    '''

    tokens = sent_token

    w2v_list = []

    for token in tokens:
        tmp_vec = ft_model.wv[token]
        w2v_list.append(tmp_vec)
        #print(len(w2v_list))

    sent_vec = np.mean(w2v_list, axis = 0)

    return sent_vec
import numpy as np
from tqdm import tqdm
import re
from konlpy.tag import Mecab
import gensim
from gensim.models import Word2Vec

# load finetuned word2vec model
w2v_path ="/repo/course/sem21_01/youtube_summarizer/src/word_embedding/model/w2v_model.model"
w2v_model = Word2Vec.load(w2v_path)

# define mecab tokenizer
mecab = Mecab()

# define stopwords
stopwords = ['의','은','는','이','가','좀','잘','과','도','을','를','으로','자','에','와','한','하다','합니다', '입니다','습니다']

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
        tmp_vec = w2v_model.wv[token]
        w2v_list.append(tmp_vec)
        #print(len(w2v_list))

    sent_vec = np.mean(w2v_list, axis = 0)

    return sent_vec
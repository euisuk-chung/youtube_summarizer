# load finetuned word2vec model

## where to locate pretrained FastText
~~~
fname = '/repo/course/sem21_01/youtube_summarizer/src/word_embedding/model/fasttext.model' # ur loc

ft_model = FastText.load(fname)
~~~

## run <sent_tokenizer.py> to tokenize and get sentence embedding vector
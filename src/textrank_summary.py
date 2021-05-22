import sys
import os
import glob
import argparse
import json
import kss
from konlpy.tag import Komoran
# from textrank import KeywordSummarizer
from textrank import KeysentenceSummarizer
# import KeysentenceSummarizer

os.path.abspath('/home/ys/repo/youtube-summarization')

def load_label(file_index):
    file_name = os.listdir(args.label_path)[file_index]
    label_file = os.path.join(args.label_path, file_name)

    with open(label_file, encoding = 'utf-8') as f:
        script = json.load(f)
        script = script['text']

    return file_name, script


def make_pos_script(file_name, script):
    list_temp = []
    for sent in kss.split_sentences(script):
        list_temp.append(sent)

    pos_script_name = './textrank/[POS]' + file_name

    komoran = Komoran()
    with open(pos_script_name, mode='wb') as output:
        for content in list_temp:
            for morph in komoran.pos(content):
                data = morph[0]+'/'+morph[1]+' '
                data = data.encode("utf-8")
                output.write(data)
            output.write('\n'.encode("utf-8"))

    print('pos tagged script saved!')
    return pos_script_name, list_temp


def prepare_data(file_index):
    file_name, script = load_label(file_index)

    pos_script_name, list_temp = make_pos_script(file_name, script)

    with open(pos_script_name, encoding='utf-8') as f:
        sents = [sent.strip() for sent in f]

    with open(os.path.join(args.label_path,file_name), encoding = 'utf-8') as f:
        texts = [sent.strip() for sent in f]

    texts = list_temp
    print('pos tagged sentence: {}, original sentence: {}'.format(len(sents), len(texts)))

    return sents, texts


def komoran_tokenize(sent):
    words = sent.split()
    words = [w for w in words if ('/NN' in w or '/XR' in w or '/VA' in w or '/VV' in w)]
    return words




if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--label_path', default = 'data/label')

    args = parser.parse_args()


    for file_index, file_name in enumerate(os.listdir(args.label_path)):
        sents, texts = prepare_data(file_index)

        summarizer = KeysentenceSummarizer(
            tokenize = komoran_tokenize,
            min_sim = 0.5,
            verbose = True,
        )

        keysents = summarizer.summarize(sents)
        
        result_file = './textrank/[Result]' + file_name + '.txt'
        with open(result_file,'ab') as f:
            for idx, rank, komoran_sent in keysents:
                result_log = '#{} ({:.3}) : {} \n'.format(idx, rank, texts[idx])
                # print(result_log)
                result_log = result_log.encode("utf-8")
                f.write(result_log)

    # for f in glob.glob(".txt"):
    #     print(f)
    #     if '[POS]' in f.item():
    #         print('start delete')
    #         os.remove(f)
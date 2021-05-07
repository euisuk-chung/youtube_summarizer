import os
import sys
import numpy as np
import random
from tqdm import tqdm
import time
import pickle
import json
import yaml
import logging 
import argparse
import torch
import re 

print(torch.__version__)

from utils.load_bertsum import bertsum
from src.backbone import WindowEmbedder
from model.subtext_classifier import SubtextClassifier


if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
    


def get_args(parser):
    return parser.parse_args()


def parse_args(parser):
    args = get_args(parser)
    y_dict = load_config(config_path=args.config_path)
    arg_dict = args.__dict__
    for key, value in y_dict.items():
        arg_dict[key] = value
    return args
        

def load_config(config_path='./config.yml'):
    with open(config_path) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    return configs


def load_jsonl(input_path) -> list:
    """
    Read list of objects from a JSON lines file.
    """
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.rstrip('\n|\r')))
    print('Loaded {} records from {}'.format(len(data), input_path))
    return data


def load_article(dataset_base_pth=''):
    
    data_path = os.path.join(dataset_base_pth, 'article_dataset/train.jsonl')
    news_df = load_jsonl(data_path)

    # 전처리
    # (1) 문장이 적은 경우 해당 기사 없애기 (10문장 이상)
    news_clean = []
    for news in news_df:
        news_article = news['article_original']
        if len(news_article) >= 10:
            article_clean = [sent for sent in news_article]
            news_clean.append(article_clean)
            
    return news_clean


def make_mixed_doc(news_dataset=None, max_num=1000):
    mixed_doc_set = []
    for i in range(max_num):
        lh_count = min(random.randint(7, 10), len(news_dataset[i]))
        rh_count = min(random.randint(7, 10), len(news_dataset[i+1]))

        lh_news = news_dataset[i][:lh_count]
        rh_news = news_dataset[i+1][:rh_count]
        
        gt = lh_count - 1

        src_doc = '\n'.join((lh_news + rh_news))
        mixed_doc_set.append((src_doc, gt))
        
    return mixed_doc_set


def get_divscore(src_doc=[], embedder=None, divider=None):
    embedding = embedder.get_embeddings(src_doc).transpose(1, 0).unsqueeze(0)
    score = divider(embedding).item()
    return score


def evaluate_divider(testset, subtext_model, embedder, window_size):
    err_cnt = 0
    acc_cnt = 0
    ws = window_size

    div_result = []
    for i, a_set in enumerate(tqdm(testset)):

        if (i+1) % 20 == 0:
            logger.info(f"working on {i+1}th doc: Accuracy so far is {acc_cnt/(acc_cnt+err_cnt)*100:.2f}%")

        src_doc = a_set[0].split('\n')
        gt = a_set[1]

        cands = [src_doc[i:i+ws*2] for i, _ in enumerate(src_doc) if i <= len(src_doc) - ws*2]

        # 가끔 한문장이 너무길어서 잘리는 경우가 있음 --> Pass
        try:
            div_scores = [get_divscore(src_doc=cand, embedder=embedder, divider=subtext_model) for cand in cands]
            div_point = div_scores.index(max(div_scores)) + ws - 1

            if div_point == gt:
                acc_cnt += 1
            else:
                err_cnt += 1

        except RuntimeError as e:
            logger.info(f"Error occurred at {i}th article... passing evaluation.")
            
    logger.info(f"Evaluation Result: {acc_cnt/(acc_cnt+err_cnt)*100:.2f}%")
    return


def create_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset_basedir", default='/home/sks/korea_univ/21_1/TA/team_project/youtube_summarizer/dataset', type=str)
    parser.add_argument("--ckpt_dir", default='/home/sks/korea_univ/21_1/TA/team_project/youtube_summarizer/src/subtext/ckpt', type=str)
    parser.add_argument("--ckpt_filename", default='subtext_model_w4.pt', type=str)
    parser.add_argument("--config_path", default='./config.yml', type=str)
    parser.add_argument("--testset_size", default=1000, type=int)

    return parser


def main():
    # logger
    global logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", 
                        level=logging.INFO,
                        handlers=[
                            logging.FileHandler(os.path.join("./evaluation.log")),
                            logging.StreamHandler()
                        ])
    
    args = parse_args(create_parser())
    logging.info(vars(args))
    
    # Check if model exists
    model_path = os.path.join(args.ckpt_dir, args.ckpt_filename)
    assert os.path.isfile(model_path), f"Given model weights doesn't exits: {model_path}"
    
    window_size = int(re.findall('\d+', args.ckpt_filename)[0])
    logger.info(f"Using window of size {window_size}")
    
    
    # Load bertsum model
    bertsum_model, loader = bertsum(args)
    bert_embedder = WindowEmbedder(model=bertsum_model, text_loader=loader)


    # Load subtext model and weights
    subtext_model = SubtextClassifier(window_size=window_size)

    
    subtext_model.load_state_dict(torch.load(model_path))
    subtext_model.eval()
    logger.info("[1/3] Subtext model loaded.")
    
    
    # Load test dataset
    news_dataset = load_article(dataset_base_pth=args.dataset_basedir)
    testset = make_mixed_doc(news_dataset=news_dataset, max_num=args.testset_size)
    logger.info("[2/3] Testset loaded.")
    
    # Evaluate
    evaluate_divider(testset, subtext_model, bert_embedder, window_size)
    
    
if __name__=='__main__':
    main()
import os
import sys
# path 추가
sys.path.append('/home/sks/korea_univ/21_1/TA/team_project/youtube_summarizer/src/bertsum')

import time
import torch
from tqdm import tqdm
import logging
import random
import pickle
import json
import yaml
import argparse
import IPython
#print(torch.__version__)

from src.backbone import WindowEmbedder
from utils.load_bertsum import bertsum



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



def clean_news(news_dataset):
    clean_set = []
    for news in news_dataset:
        news_article = news['article_original']
        if len(news_article) >= 10:
            article_clean = [sent for sent in news_article]
            clean_set.append(article_clean)
    return clean_set


def load_article(dataset_base_pth='', dataset_size=50000):
    
    trainset_path = os.path.join(dataset_base_pth, 'article_dataset/train.jsonl')
    devset_path = os.path.join(dataset_base_pth, 'article_dataset/dev.jsonl')
    testset_path = os.path.join(dataset_base_pth, 'article_dataset/test.jsonl')
    
    news_df_train = clean_news(load_jsonl(trainset_path))
    news_df_dev = clean_news(load_jsonl(devset_path))
    news_df_test = clean_news(load_jsonl(testset_path))
    
    train_len, dev_len = int(dataset_size*0.8), int(dataset_size*0.2)
    news_df = news_df_train[:train_len] + news_df_dev[:dev_len]

    return news_df, news_df_test


# def bertsum_model(configs):
#     # Settings
#     device = "cpu" if configs.visible_gpus == -1 else "cuda"
#     loader = TextLoader(configs, device)

#     # model setting
#     ckpt_path = '/home/sks/korea_univ/21_1/TA/team_project/youtube_summarizer/src/bertsum/checkpoint/model_step_24000.pt'
#     checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
#     model = ExtSummarizer(configs, device, checkpoint)
#     model.eval()
    
#     return model, loader


def save_data(args, file):
    random_flag = 'random' if args.random_point else 'fixed'
    
    if args.embed_type == 'bert':
        save_path = os.path.join(args.dataset_basedir, f'subtext_dataset/nn_dataset_w{args.window_size}_{random_flag}.pkl')
    elif args.embed_type == 'word':
        save_path = os.path.join(args.dataset_basedir, f'subtext_dataset/nn_dataset_w2v_w{args.window_size}_{random_flag}.pkl')
    else: 
        raise ValueError('Wrong input for embed_type')
        
    with open(save_path, 'wb') as ww:
        pickle.dump(file, ww)
        print(f"Saving done at: {save_path}")
        
    return save_path



class DataGenerator:
    '''
    Information: Generate dataset for training 1d-conv model.
    Arguments
        max_num: Total length of dataset(articles). Dataset is generated at ratio of 8:2 for train.jsonl and dev.jsonl each
                 example) if 50000, 40000 is from train.jsonl and 10000 is from dev.jsonl
    '''
    def __init__(self, news_dataset=None, news_testset=None, window_size=3, y_ratio=0.5, random_point=False):
        
        self.df_base = news_dataset
        self.df_test = news_testset
        self.window_size = window_size
        self.y_ratio = y_ratio
        self.random_point = random_point
        logger.info(f"Generating data based on random points") if random_point else logger.info(f"Generating data based on fixed points")
        
    def make_base(self):
        
        random.seed(1234)
        random.shuffle(self.df_base)
        
        train_ratio = 0.7
        val_ratio = 0.3

        train_len, val_len = int(len(self.df_base)*train_ratio), int(len(self.df_base)*val_ratio)
        test_len = len(self.df_test)
        print(f"Train_base: {train_len}, Val_base: {val_len}, Test_base: {test_len}")

        train_base = self.df_base[:train_len]
        val_base = self.df_base[train_len:]
        
        return train_base, val_base

    
    def dataset_generator(self, base_dataset=None, y_ratio=0.5):
        '''
        y_data: Mixed article
        n_data: Normal article
        '''
        random_point = self.random_point
        window_size = self.window_size
        tot_len = base_dataset.__len__()
        y_len = int(tot_len * y_ratio)

        y_cands = base_dataset[:y_len]
        n_cands = base_dataset[y_len:]

        y_dataset, n_dataset = [], []
        for i in tqdm(range(len(y_cands) - 1), desc='Sampling mixed dataset'):
            
            if not random_point:
                start_i = 0
                start_j = 0
            else:
                # random start point generator
                start_i = random.randint(0, (len(y_cands[i]) - window_size))
                start_j = random.randint(0, (len(y_cands[i+1]) - window_size))
            
            tmp_article_y = y_cands[i][start_i:start_i+window_size] + y_cands[i+1][start_j:start_j+window_size]
            y_dataset.append(tmp_article_y)

        for j in tqdm(range(len(n_cands)), desc='Sampling normal dataset'):
            
            if not random_point:
                start_ii = 0
            else:
                start_ii = random.randint(0, (len(n_cands[j]) - window_size*2))
                
            tmp_article_n = n_cands[j][start_ii:start_ii+window_size*2]
            n_dataset.append(tmp_article_n)

        return y_dataset, n_dataset

    
    def make_data(self):
        
        train_base, val_base = self.make_base()
        test_base = self.df_test
        
        train_div, train_org = self.dataset_generator(base_dataset=train_base, y_ratio=self.y_ratio)
        val_div, val_org = self.dataset_generator(base_dataset=val_base, y_ratio=self.y_ratio)
        test_div, test_org = self.dataset_generator(base_dataset=test_base, y_ratio=self.y_ratio)
        
        return train_div, train_org, val_div, val_org, test_div, test_org

    
    def make_tensor(self, embedder=None):
        
        window_size = self.window_size
        
        tot_datasets = self.make_data()
        tot_embs = []
        for dt in tot_datasets:
            tmp_emb = []
            # changed as of 21.05.22
            for article in tqdm(dt, desc="working on articles"):
                embedding = embedder.get_embeddings(article)
                if (embedding is not None) and (embedding.size()[0] == window_size*2):
                    tmp_emb.append(embedding.unsqueeze(0))
            tmp_dt_emb = torch.cat(tmp_emb, dim=0)
            tot_embs.append(tmp_dt_emb)
            
        return tot_embs


def create_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset_basedir", default='/home/sks/korea_univ/21_1/TA/team_project/youtube_summarizer/dataset', type=str)
    #/repo/course/sem21_01/youtube_summarizer/dataset
    parser.add_argument("--config_path", default='./config.yml', type=str)
    parser.add_argument("--window_size", default=4, type=int)
    parser.add_argument("--dataset_size", default=50000, type=int)
    parser.add_argument("--random_point", action='store_true')
    parser.add_argument("--embed_type", default='bert', type=str, help='[bert, word]')

    return parser
    

def main():
    
    global logger
    # logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", 
                        level=logging.INFO,
                        handlers=[
                            logging.FileHandler(os.path.join("./make_data.log")),
                            logging.StreamHandler()
                        ])
    
    args = parse_args(create_parser())
    logging.info(vars(args))
    logging.info(f"Generate using random points: {args.random_point}")
    
    # Load bertsum model and embedder
    if args.embed_type == 'bert':
        bertsum_model, loader = bertsum(args)
        embedder = WindowEmbedder(model=bertsum_model, text_loader=loader, embed_type=args.embed_type)
    else:
        embedder = WindowEmbedder(model=None, text_loader=None, embed_type=args.embed_type)
    logger.info(f"[1/4] Embedder loaded. Using {args.embed_type} embedding")
    
    # Load article dataset
    news_dataset, news_testset = load_article(dataset_base_pth=args.dataset_basedir, dataset_size=args.dataset_size)
    logger.info(f"[2/4] Dataset loaded.")
    
    # Make tensor dataset used for training subtext_nn model
    data_generator = DataGenerator(news_dataset=news_dataset,
                                   news_testset=news_testset,
                                   window_size=args.window_size,
                                   random_point=args.random_point)
    
    tensor_dataset = data_generator.make_tensor(embedder=embedder)
    logger.info(f"[3/4] Making tensors finished.")
    
    # Save tensor dataset into .pkl file at given directory
    save_path = save_data(args, tensor_dataset)
    logger.info(f"[4/4] Saving done at {save_path}")
    
    
if __name__=='__main__':
    main()
import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
import time
import yaml
import pickle
import json
import logging

import torch
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
    
# custom
from utils.load_bertsum import bertsum
from utils.preprocess import doc_preprocess
from src.backbone import WindowEmbedder
from model.subtext_classifier import SubtextClassifier



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


def load_json(input_path):
    with open(input_path, 'rb') as rr:
        json_file = json.load(rr)
    return json_file


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


def get_prob(logit):
    prob = 1/(1+np.exp(-logit))
    return prob


class SubtextDivider:
    
    def __init__(self, embedder=None, script_pth='', window_list=[]):
        
        self.embedder = embedder
        self.script_pth = script_pth
        self.window_list = window_list
        self.script_list = self.load_youtube_script(filename=script_pth)


    def load_youtube_script(self, filename='KBS뉴스_7_XpWIWY6pQ_27m_51s.txt'):
        youtube_script_pth = os.path.join('/home/sks/korea_univ/21_1/TA/team_project/youtube_summarizer/dataset/youtube_dataset/label', filename)
        assert os.path.isfile(youtube_script_pth), f"No such script file exists: {youtube_script_pth}"

        youtube_df = load_json(youtube_script_pth)

        script = youtube_df['text']
        script_fin = doc_preprocess(script) # preprocess on script

        script_list = [sent for sent in script_fin.split('\n') if len(sent.strip()) >= 20]#[:50]
        
        return script_list


    def get_mean_scores(self, embedder=None):
        '''
        Information
            ddd
        Arguments
            embedder: Bertsum embedder to get embeddings of given sentences.
            subtext_model: Model for sub-texting the given script.
            window_size: List of window sized where each component is used for dividing and getting div_scores.
        '''
        assert self.window_list, "Give window size!"
        
        window_list = self.window_list
        
        fin_scoreset = []
        for ws in window_list:

            # Load subtext model of window size
            subtext_model = SubtextClassifier(window_size=ws).to(device)
            model_path = f'/home/sks/korea_univ/21_1/TA/team_project/youtube_summarizer/src/subtext/ckpt/subtext_model_w{ws}_fixed.pt'
            subtext_model.load_state_dict(torch.load(model_path))
            subtext_model.eval()

            # Load youtube script data(list)
            script_list = self.script_list

            # Base numpy array for saving scores
            base = np.zeros(len(script_list)-1)

            # Get embeddings and calculate scores
            score_list = []
            offset = ws-1
            for i in tqdm(range(len(script_list)-ws*2+1), desc=f"window size={ws} "):
                w_input = script_list[i:i+(ws*2)]

                # embedding
                emb = self.embedder.get_embeddings(w_input).transpose(1, 0).to(device)
                score = subtext_model(emb.unsqueeze(0)).item()
                score_list.append(score)

            base[offset:len(base)-offset] = np.array(score_list)
            fin_scoreset.append(base)

        return np.mean(np.array(fin_scoreset), axis=0)


    def write_subtexts(self, output_pth='./results/tmp.txt'):
        
        # load script
        script_list = self.script_list
        
        # load score
        div_score = self.get_mean_scores(embedder=self.embedder)
        
        # write
        with open(output_pth, 'w') as file:
            i = 0
            keep_flag = True
            while keep_flag:
                to_print = f"{script_list[i]}\n\n====== {div_score[i]:.2f} =====\n" if div_score[i] > 0 else f"{script_list[i]}\n{div_score[i]:.2f}"
                file.write(to_print)
                i += 1
                keep_flag = False if i == len(div_score) else True
            file.close()
        return



def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default='./config.yml', type=str)
    parser.add_argument("--script_pth", default='KBS뉴스_7_XpWIWY6pQ_27m_51s.txt', type=str)
    parser.add_argument("--window_list", metavar='N', type=int, nargs='+', default='', help='Integers with space. ex) 1 2 3 4')
    parser.add_argument("--output_pth", default='./results/tmp.txt', type=str)

    return parser


def main():
    # logger
    global logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", 
                        level=logging.INFO,
                        handlers=[
                            logging.FileHandler(os.path.join("./subtext_writer.log")),
                            logging.StreamHandler()
                        ])
    
    args = parse_args(create_parser())
    logging.info(vars(args))
    
    
    # Load bertsum model
    bertsum_model, loader = bertsum(args)
    embedder = WindowEmbedder(model=bertsum_model, text_loader=loader)
    logger.info(f"[1/3] Bertsum model loaded.")
    
    divider = SubtextDivider(embedder=embedder, script_pth=args.script_pth, window_list=args.window_list)
    logger.info(f"[2/3] Subtext dividing model loaded.")
    
    # Write result sub-texted script
    divider.write_subtexts(output_pth=args.output_pth)
    logger.info(f"[3/3] Sub-texting Finished.")
    
    
if __name__=='__main__':
    main()
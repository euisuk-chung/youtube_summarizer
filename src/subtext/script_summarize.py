import os
import sys
import argparse
import re
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
from src.backbone import WindowEmbedder, Extractor
from src.summarize import SubtextSummarizer
from model.subtext_classifier import SubtextClassifier


import IPython


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
    
    def __init__(self, args=None, embedder=None, script_pth='', window_list=[], threshold=0.0, mode='mean'):
        
        self.args = args
        self.embedder = embedder
        self.script_pth = script_pth
        self.window_list = window_list
        self.script_list, self.script_segs = self.load_youtube_script(filename=script_pth)
        self.summary_result = None
        
        self.threshold = threshold
        self.mode = mode


    def load_youtube_script(self, filename='KBS뉴스_7_XpWIWY6pQ_27m_51s.txt'):
        youtube_script_pth = os.path.join('/home/sks/korea_univ/21_1/TA/team_project/youtube_summarizer/dataset/youtube_dataset/label', filename)
        assert os.path.isfile(youtube_script_pth), f"No such script file exists: {youtube_script_pth}"

        youtube_df = load_json(youtube_script_pth)

        script = youtube_df['text']
        script_segs = [seg['words'] for seg in youtube_df['segments'] if seg['textEdited']]
        script_fin = doc_preprocess(script) # preprocess on script
        script_list = [sent+'.' for sent in script_fin.split('\n') if len(sent.strip()) >= 20]#[:50]
        
        return script_list, script_segs
    
    

    def _rule_refine(self, score_input, window_size):
        '''
        Info: 산출된 스코어 리스트를 특정 룰에 기반하여 정제하는 과정
              window_size=3인 경우 3개 스코어를 연속으로 봤을 때, 가장 큰 값만 분리지점으로 판단
        '''
        score_list = score_input.copy()
        for i in range(0, len(score_list)-window_size+1):
            cands = np.array(score_list[i:i+window_size])
            tmp = np.where(
                (cands!=max(cands)) & (cands >= 0),
                -10,
                cands
            )
            score_list[i:i+window_size] = tmp
        return score_list

    
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

        
        if self.mode == 'mean':
            mean = np.mean(np.array(fin_scoreset), axis=0)
            mean_refined = self._rule_refine(mean, 3)
            mean_score = np.where(mean_refined >= self.threshold, 1, 0)
        else:
            vote = np.mean(np.where(np.array(fin_scoreset) >= self.threshold, 1, 0), axis=0)
            mean_score = np.where(vote >= 0.5, 1, 0)

        return mean_score

    
    
    def _divider(self, script_list, div_idx):
        '''
        Return the script list divided by division scores.
        '''
        div_points = div_idx + 1
        tmp_txt = script_list.copy()
        result_list = []
        
        curr_idx = 0
        div_sub = 0
        for i, div in enumerate(div_points):
            curr_idx = div if i == 0 else div - div_sub
            
            div_lh, div_rh = tmp_txt[:curr_idx], tmp_txt[curr_idx:]
            result_list.append(div_lh)

            if i == len(div_idx)-1:
                result_list.append(div_rh)

            tmp_txt = div_rh
            div_sub = div
            
        return result_list
    
    
    def get_subtexts(self, save=True, output_pth='./results/tmp.txt'):
        
        # load script
        script_list = self.script_list
        
        # load score
        div_score = self.get_mean_scores(embedder=self.embedder)
        div_idx = np.ravel(np.argwhere(div_score == 1))
        div_result = self._divider(script_list, div_idx)
        
        # Summarize each subtext
        summarizer = SubtextSummarizer(args=self.args, ckpt_path=self.args.bertsum_weight, input_script=div_result)
        summary_result = summarizer.summarize_subtexts()
        self.summary_result = summary_result
        
        if save:
            # write
            with open(output_pth, 'w') as file:
                for subtext, summary in zip(div_result, summary_result):
                    
                    subtext_tmp = '\n'.join(subtext)
                    subtext_print = f"[ Original ]\n{subtext_tmp}"
                    
                    summary_tmp = '\n'.join(summary)
                    summary_print = f"[ Summary ]\n{summary_tmp}"
                    
                    to_print = f"\n{subtext_print}\n{summary_print}\n\n\n"
                    file.write(to_print)
                
        return div_result
    
    
    def match_time(self, output_pth='./results/tmp_time.txt'):
        assert self.summary_result, "No summary result has been made."
        summary_result = self.summary_result
        script_segs = self.script_segs

        words_set = [item for sublist in script_segs for item in sublist]
        words = np.array([re.sub('([^\d])(\.)', lambda m: "{}".format(m.group(1)), aset[-1]) for aset in words_set], dtype=object)
        
        first_summaries = [summ[0] for summ in summary_result]
        
        match_result = []
        for summ_idx, summary in enumerate(first_summaries):
            keys = summary.split(' ')

            match_score = []
            for key in keys:
                match_list = np.where(words==key, 1, 0)
                match_score.append(match_list)

            score_list = np.max(np.array(match_score), axis=0)
            window_size = len(keys)
            
            max_val, max_idx = 0, 0
            for i in range(0, len(score_list)-window_size+1):
                tmp_max = np.sum(score_list[i:i+window_size])
                # update
                if tmp_max > max_val:
                    max_val = tmp_max
                    max_idx = i
            
            match_time = words_set[max_idx][0]
            match_result.append((match_time, summary_result[summ_idx]))

        # write
        with open(output_pth, 'w') as file:
            for mat_time, summary in match_result:
                mat_time_sec = mat_time/1000
                time_m, time_s = round(divmod(mat_time_sec, 60)[0]), round(divmod(mat_time_sec, 60)[1])
                
                # write times
                file.write(f"\n[Time]: {time_m}m {time_s}s\n")
                    
                # write summary
                file.write("[Summary]\n")
                for summ in summary:
                    file.write(f"{summ}\n")
        
        return match_result
        
        



def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default='./config.yml', type=str)
    parser.add_argument("--script_pth", default='KBS뉴스_7_XpWIWY6pQ_27m_51s.txt', type=str)
    parser.add_argument("--window_list", metavar='N', type=int, nargs='+', default='', help='Integers with space. ex) 1 2 3 4')
    parser.add_argument("--mode", default='mean', help='Type to decide division points. [mean, vote]')
    parser.add_argument("--threshold", default=0.0, type=float)
    parser.add_argument("--save_result", action='store_true')
    parser.add_argument("--output_pth", default='./results/tmp.txt', type=str)
    parser.add_argument("--bertsum_weight", default='/home/sks/korea_univ/21_1/TA/team_project/youtube_summarizer/src/bertsum/checkpoint/model_step_24000.pt')
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
    logger.info(f"[1/4] Bertsum model loaded.")
    
    divider = SubtextDivider(args=args, embedder=embedder, script_pth=args.script_pth, window_list=args.window_list, threshold=args.threshold)
    logger.info(f"[2/4] Subtext dividing model loaded.")
    
    # Write result sub-texted script
    script_list = divider.get_subtexts(save=args.save_result, output_pth=args.output_pth)
    logger.info(f"Save to .txt file: {args.save_result}")
    logger.info(f"[3/4] Finished summarizing subtexts.")
    
    # Summarize each subtext
    time_output = args.output_pth.split('.txt')[0] + '_time.txt'
    summary_match = divider.match_time(output_pth=time_output)
    logger.info(f"[4/4] Matching times done.")
    
    
if __name__=='__main__':
    main()
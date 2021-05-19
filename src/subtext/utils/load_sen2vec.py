import os
import sys

# path 추가
sys.path.append('/repo/course/sem21_01/youtube_summarizer/src/word_embedding/') 

from sent_tokenizer import get_sent_token, get_sent_embedding


def sen2vec(sentence):
    # # Settings
    # device = "cpu" if configs.visible_gpus == -1 else "cuda"
    # loader = TextLoader(configs, device)

    # # model setting
    # ckpt_path = '/repo/course/sem21_01/youtube_summarizer/src/bertsum/checkpoint/model_step_24000.pt' #의석
    # # '/home/sks/korea_univ/21_1/TA/team_project/youtube_summarizer/src/bertsum/checkpoint/model_step_24000.pt' #규성
    # checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    # model = ExtSummarizer(configs, device, checkpoint)
    # model.eval()

    # for model structure
    loader = None
    
    # get tokens from given sentence
    sent_token = get_sent_token(sentence)

    # get sentence embedding vector
    sent_vec = get_sent_embedding(sent_token)

    return sent_vec

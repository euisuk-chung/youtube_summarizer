import os
import sys
# path 추가
sys.path.append('/home/sks/korea_univ/21_1/TA/team_project/youtube_summarizer/src/bertsum')
import torch

from models.data_loader import TextLoader, load_dataset
from src.backbone import ExtTransformerEncoder, ExtSummarizer, WindowEmbedder


def bertsum(configs):
    # Settings
    device = "cpu" if configs.visible_gpus == -1 else "cuda"
    loader = TextLoader(configs, device)

    # model setting
    ckpt_path = '/home/sks/korea_univ/21_1/TA/team_project/youtube_summarizer/src/bertsum/checkpoint/model_step_24000.pt'
    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    model = ExtSummarizer(configs, device, checkpoint)
    model.eval()
    
    return model, loader



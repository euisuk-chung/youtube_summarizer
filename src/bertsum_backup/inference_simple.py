from models.predictor import build_predictor
from models.model_builder import AbsSummarizer, ExtSummarizer
from kobert.utils import get_tokenizer
from models.data_loader import TextLoader, load_dataset, Dataloader, get_kobert_vocab
from models.trainer_ext import build_trainer
import gluonnlp as nlp
from easydict import EasyDict
import os
import torch

args = EasyDict({
    "visible_gpus" : -1,
    "temp_dir" : './tmp/',
    "test_from": None,
    "max_pos" : 512,
    "large" : False,
    "finetune_bert": True,
    "encoder": "bert",
    "share_emb": False,
    "dec_layers": 6,
    "dec_dropout": 0.2,
    "dec_hidden_size": 768,
    "dec_heads": 8,
    "dec_ff_size": 2048,
    "enc_hidden_size": 512,
    "enc_ff_size": 512,
    "enc_dropout": 0.2,
    "enc_layers": 6,
    
    "ext_dropout": 0.2,
    "ext_layers": 2,
    "ext_hidden_size": 768,
    "ext_heads": 8,
    "ext_ff_size": 2048,
    
    "accum_count": 1,
    "save_checkpoint_steps": 5,
    
    "generator_shard_size": 32,
    "alpha": 0.6,
    "beam_size": 5,
    "min_length": 15,
    "max_length": 150,
    "max_tgt_len": 140,  
    "block_trigram": True,
    
    "model_path": "./tmp_model/",
    "result_path": "./tmp_result/src",
    "recall_eval": False,
    "report_every": 1,
})

class Extractor:
    def __init__(self, use_gpu, checkpoint_path):
        args.test_from = checkpoint_path
        os.makedirs(args.temp_dir, exist_ok=True)
        os.makedirs(args.model_path, exist_ok=True)
        os.makedirs(os.path.split(args.result_path)[0], exist_ok=True)

        model_flags = ['hidden_size', 'ff_size', 'heads', 'inter_layers', 'encoder', 'ff_actv', 'use_interval', 'rnn_size']

        args.visible_gpus = 0 if use_gpu else -1
        device = "cpu" if args.visible_gpus == -1 else "cuda"
        device_id = 0 if device == "cuda" else -1
        args.world_size = 1
        args.gpu_ranks = [0]
        
        checkpoint = torch.load(args.test_from, map_location=lambda storage, loc: storage)

        opt = vars(checkpoint['opt'])
        for k in opt.keys():
            if (k in model_flags):
                setattr(args, k, opt[k])

        model = ExtSummarizer(args, device, checkpoint)
        model.eval()

        #vocab = get_kobert_vocab(cachedir=args.temp_dir)
        
        self.predictor = build_trainer(args, device_id, model, None)
#         self.loader = Dataloader(args, load_dataset(args, 'test', shuffle=False),
#                                        1, device,
#                                        shuffle=False, is_test=True)
        self.loader = TextLoader(args, device)
    
    def summarize(self, src, delimiter=None):
        test_iter = self.loader.load_text(src, delimiter)
        return self.predictor.test(test_iter, step=0, return_results=True)

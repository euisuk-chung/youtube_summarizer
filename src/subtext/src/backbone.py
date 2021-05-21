import os
import numpy as np 
import math
import torch
import torch.nn as nn
import sys

sys.path.append('/home/sks/korea_univ/21_1/TA/team_project/youtube_summarizer/src/bertsum') #규성
#sys.path.append('/repo/course/sem21_01/youtube_summarizer/src/bertsum') #의석

from models.predictor import build_predictor
from kobert.utils import get_tokenizer
from models.trainer_ext import build_trainer, Trainer
from models.neural import MultiHeadedAttention, PositionwiseFeedForward
from models.model_builder import Bert
from models.encoder import Classifier, PositionalEncoding, TransformerEncoderLayer, ExtTransformerEncoder
from models.data_loader import TextLoader, load_dataset, Dataloader, get_kobert_vocab
#from utils.load_sen2vec import sen2vec
import gluonnlp as nlp
import IPython



class Extractor:
    def __init__(self, args, use_gpu, checkpoint_path):
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
        
        self.predictor = build_trainer(args, device_id, model, None)
        self.loader = TextLoader(args, device)
    
    def summarize(self, src, delimiter=None):
        test_iter = self.loader.load_text(src, delimiter)
        return self.predictor.test(test_iter, step=0, return_results=True)

    
    
class ExtTransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout, num_inter_layers=0):
        super(ExtTransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_inter_layers = num_inter_layers
        self.pos_emb = PositionalEncoding(dropout, d_model)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_inter_layers)])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.wo = nn.Linear(d_model, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, top_vecs, mask):
        """ See :obj:`EncoderBase.forward()`"""

        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        pos_emb = self.pos_emb.pe[:, :n_sents]
        x = top_vecs * mask[:, :, None].float()
        x = x + pos_emb

        for i in range(self.num_inter_layers):
            x = self.transformer_inter[i](i, x, x, 1 - mask)  # all_sents * max_tokens * dim

        x = self.layer_norm(x)
        sent_scores = self.sigmoid(self.wo(x))
        sent_scores = sent_scores.squeeze(-1) * mask.float()

        return sent_scores

    
class ExtSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint):
        super(ExtSummarizer, self).__init__()
        self.device = device
        self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)

        self.ext_layer = ExtTransformerEncoder(self.bert.model.config.hidden_size, args.ext_ff_size, args.ext_heads,
                                               args.ext_dropout, args.ext_layers)
        if (args.encoder == 'baseline'):
            bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.ext_hidden_size,
                                     num_hidden_layers=args.ext_layers, num_attention_heads=args.ext_heads, intermediate_size=args.ext_ff_size)
            self.bert.model = BertModel(bert_config)
            self.ext_layer = Classifier(self.bert.model.config.hidden_size)

        if(args.max_pos>512):
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings


        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            if args.param_init != 0.0:
                for p in self.ext_layer.parameters():
                    p.data.uniform_(-args.param_init, args.param_init)
            if args.param_init_glorot:
                for p in self.ext_layer.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)

        self.to(device)

    def forward(self, src, segs, clss, mask_src, mask_cls):
        top_vec = self.bert(src, segs, mask_src)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        sent_scores = self.ext_layer(sents_vec, mask_cls).squeeze(-1)
        return sents_vec, sent_scores, mask_cls
    
    
class WindowEmbedder:
    def __init__(self, model=None, text_loader=None, embed_type='bert'):
        self.text_loader = text_loader
        self.model = model
        self.embed_type = embed_type

    def embedder(self, target_doc=None):
        model = self.model
        batch_iter = self.text_loader.load_text(target_doc, '\n')
        for _, batch in enumerate(batch_iter):
            src = batch.src
            segs = batch.segs
            clss = batch.clss
            mask, mask_cls = batch.mask_src, batch.mask_cls
            result_vec, _, _ = model(src, segs, clss, mask, mask_cls)
        return result_vec.detach()

    def word_embedder(self, target_doc=None):
        result_vec = [sen2vec(sent) for sent in target_doc]
        return result_vec
    
    def get_embeddings(self, sents):
        embed_type = self.embed_type
        if embed_type == 'bert':
            target_doc = '\n'.join(sents)
            tmp_embedded = self.embedder(target_doc=target_doc)
            tmp_embedded = tmp_embedded.squeeze(0)
        elif embed_type == 'word':
            target_doc = sents
            try:
                tmp_embedded = torch.tensor(self.word_embedder(target_doc=target_doc))
            except TypeError as e:
                print("Embedding failed. Passing")
                return None
        return tmp_embedded
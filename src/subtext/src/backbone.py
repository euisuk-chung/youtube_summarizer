import os
import sys
import numpy as np 
import math
import copy

import torch
import torch.nn as nn

sys.path.append('/home/sks/korea_univ/21_1/TA/team_project/youtube_summarizer/src/bertsum') #규성
#sys.path.append('/repo/course/sem21_01/youtube_summarizer/src/bertsum') #의석

from models.predictor import build_predictor
from kobert.utils import get_tokenizer
from models.trainer_ext import build_trainer, Trainer
from models.neural import MultiHeadedAttention, PositionwiseFeedForward
from models.model_builder import Bert, get_generator
from models.encoder import Classifier, PositionalEncoding, TransformerEncoderLayer, ExtTransformerEncoder
from models.decoder import TransformerDecoderLayer, TransformerDecoderState
from models.data_loader import TextLoader, load_dataset, Dataloader, get_kobert_vocab
#from utils.load_sen2vec import sen2vec
import gluonnlp as nlp
import IPython





# --------------------------------------------
#    Window Embedder using Ext Summarizer
# --------------------------------------------

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





# ---------------------------------------------
#     Extractor: Extractive Summarization
# ---------------------------------------------

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
        
#         if (args.encoder == 'baseline'):
#             bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.ext_hidden_size,
#                                      num_hidden_layers=args.ext_layers, num_attention_heads=args.ext_heads, intermediate_size=args.ext_ff_size)
#             self.bert.model = BertModel(bert_config)
#             self.ext_layer = Classifier(self.bert.model.config.hidden_size)

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
    
    

    
    
    
    
# ---------------------------------------------
#     Generator: Abstractive Summarization
# ---------------------------------------------

class Generator:
    def __init__(self, args, use_gpu, checkpoint_path):
        args.test_from = checkpoint_path
        os.makedirs(args.temp_dir, exist_ok=True)
        os.makedirs(args.model_path, exist_ok=True)
        os.makedirs(os.path.split(args.result_path)[0], exist_ok=True)

        model_flags = ['hidden_size', 'ff_size', 'heads', 'emb_size', 'enc_layers', 'enc_hidden_size', 'enc_ff_size',
               'dec_layers', 'dec_hidden_size', 'dec_ff_size', 'encoder', 'ff_actv', 'use_interval']

        args.visible_gpus = 0 if use_gpu else -1
        device = "cpu" if args.visible_gpus == -1 else "cuda"
        checkpoint = torch.load(args.test_from, map_location=lambda storage, loc: storage)

        opt = vars(checkpoint['opt'])
        for k in opt.keys():
            if (k in model_flags):
                setattr(args, k, opt[k])

        model = AbsSummarizer(args, device, checkpoint)
        model.eval()

        vocab = get_kobert_vocab(cachedir=args.temp_dir)
        #tokenizer = nlp.data.BERTSPTokenizer(get_tokenizer(), vocab, lower=False)
        symbols = {'BOS': vocab.token_to_idx['[BOS]'], 'EOS': vocab.token_to_idx['[EOS]'],
                'PAD': vocab.token_to_idx['[PAD]'], 'EOQ': vocab.token_to_idx['[EOS]']}
        self.predictor = build_predictor(args, vocab, symbols, model)
        self.loader = TextLoader(args, device)
    
    def summarize(self, src, num_beams=5):
        self.loader.args.num_beams = num_beams
        test_iter = self.loader.load_text(src)
        return self.predictor.translate(test_iter, step=-1, return_results=True)
    

    
class AbsSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint=None, bert_from_extractive=None):
        super(AbsSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)

        if bert_from_extractive is not None:
            self.bert.model.load_state_dict(
                dict([(n[11:], p) for n, p in bert_from_extractive.items() if n.startswith('bert.model')]), strict=True)

#         if (args.encoder == 'baseline'):
#             bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.enc_hidden_size,
#                                      num_hidden_layers=args.enc_layers, num_attention_heads=8,
#                                      intermediate_size=args.enc_ff_size,
#                                      hidden_dropout_prob=args.enc_dropout,
#                                      attention_probs_dropout_prob=args.enc_dropout)
#             self.bert.model = BertModel(bert_config)

        if(args.max_pos>512):
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings
            
        self.vocab_size = self.bert.model.config.vocab_size
        tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
        
        if (self.args.share_emb):
            tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)

        self.decoder = TransformerDecoder(
            self.args.dec_layers,
            self.args.dec_hidden_size, heads=self.args.dec_heads,
            d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings)

        self.generator = get_generator(self.vocab_size, self.args.dec_hidden_size, device)
        self.generator[0].weight = self.decoder.embeddings.weight


        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            for module in self.decoder.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            for p in self.generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
                else:
                    p.data.zero_()
            if(args.use_bert_emb):
                tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
                tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)
                self.decoder.embeddings = tgt_embeddings
                self.generator[0].weight = self.decoder.embeddings.weight

        self.to(device)

    def forward(self, src, tgt, segs, clss, mask_src, mask_tgt, mask_cls):
        top_vec = self.bert(src, segs, mask_src)
        dec_state = self.decoder.init_decoder_state(src, top_vec)
        decoder_outputs, state = self.decoder(tgt[:, :-1], top_vec, dec_state)
        return decoder_outputs, None
    
    

class TransformerDecoder(nn.Module):
    """
    The Transformer decoder from "Attention is All You Need".

    .. mermaid::
       graph BT
          A[input]
          B[multi-head self-attn]
          BB[multi-head src-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> BB
          BB --> C
          C --> O

    Args:
       num_layers (int): number of encoder layers.
       d_model (int): size of the model
       heads (int): number of heads
       d_ff (int): size of the inner FF layer
       dropout (float): dropout parameters
       embeddings (:obj:`onmt.modules.Embeddings`):
          embeddings to use, should have positional encodings
       attn_type (str): if using a seperate copy attention
    """
    def __init__(self, num_layers, d_model, heads, d_ff, dropout, embeddings):
        super(TransformerDecoder, self).__init__()

        # Basic attributes
        self.decoder_type = 'transformer'
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.pos_emb = PositionalEncoding(dropout,self.embeddings.embedding_dim)


        # Build TransformerDecoder
        self.transformer_layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_layers)])

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, tgt, memory_bank, state, memory_lengths=None,
                step=None, cache=None,memory_masks=None):
        """
        See :obj:`onmt.modules.RNNDecoderBase.forward()`
        """

        src_words = state.src
        tgt_words = tgt
        src_batch, src_len = src_words.size()
        tgt_batch, tgt_len = tgt_words.size()

        # Run the forward pass of the TransformerDecoder.
        # emb = self.embeddings(tgt, step=step)
        emb = self.embeddings(tgt)
        assert emb.dim() == 3  # len x batch x embedding_dim

        output = self.pos_emb(emb, step)

        src_memory_bank = memory_bank
        padding_idx = self.embeddings.padding_idx
        tgt_pad_mask = tgt_words.data.eq(padding_idx).unsqueeze(1) \
            .expand(tgt_batch, tgt_len, tgt_len)

        if (not memory_masks is None):
            src_len = memory_masks.size(-1)
            src_pad_mask = memory_masks.expand(src_batch, tgt_len, src_len)

        else:
            src_pad_mask = src_words.data.eq(padding_idx).unsqueeze(1) \
                .expand(src_batch, tgt_len, src_len)

        if state.cache is None:
            saved_inputs = []

        for i in range(self.num_layers):
            prev_layer_input = None
            if state.cache is None:
                if state.previous_input is not None:
                    prev_layer_input = state.previous_layer_inputs[i]
            output, all_input \
                = self.transformer_layers[i](
                    output, src_memory_bank,
                    src_pad_mask, tgt_pad_mask,
                    previous_input=prev_layer_input,
                    layer_cache=state.cache["layer_{}".format(i)]
                    if state.cache is not None else None,
                    step=step)
            if state.cache is None:
                saved_inputs.append(all_input)

        if state.cache is None:
            saved_inputs = torch.stack(saved_inputs)

        output = self.layer_norm(output)

        # Process the result and update the attentions.

        if state.cache is None:
            state = state.update_state(tgt, saved_inputs)

        return output, state

    def init_decoder_state(self, src, memory_bank,
                           with_cache=False):
        """ Init decoder state """
        state = TransformerDecoderState(src)
        if with_cache:
            state._init_cache(memory_bank, self.num_layers)
        return state
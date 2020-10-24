import torch
import torch.nn as nn
from torch.nn import functional as F
from fastai.torch_core import Module # Same as nn.Module but no need for subclass to call super()
from fastcore.utils import store_attr
from fastai.text.models.awdlstm import EmbeddingDropout



AWD_LSTM = dict()
    # {'hid_name':'emb_sz', 'url':URLs.WT103_FWD, 'url_bwd':URLs.WT103_BWD,
    # 'config_lm':awd_lstm_lm_config, 'split_lm': awd_lstm_lm_split,
    # 'config_clas':awd_lstm_clas_config, 'split_clas': awd_lstm_clas_split}

class AWD_LSTM(Module):
    def __init__(self, vocab_sz, emb_sz, pad_token=1):
        store_attr('emb_sz,pad_token')
        self.encoder = nn.Embedding(vocab_sz, emb_sz, padding_idx=pad_token)
        self.encoder_dp = EmbeddingDropout(self.encoder, embed_p)
        


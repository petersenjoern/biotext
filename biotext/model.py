import torch
import torch.nn as nn
from torch.nn import functional as F
from fastai.torch_core import Module # Same as nn.Module but no need for subclass to call super()
from fastcore.utils import store_attr
from fastai.text.models.awdlstm import EmbeddingDropout, WeightDropout


awd_lstm_lm_config = dict(
    emb_sz=400, n_hid=1152, n_layers=3, pad_token=1, bidir=False, output_p=0.1,
    hidden_p=0.15, input_p=0.25, embed_p=0.02, weight_p=0.2, tie_weights=True, out_bias=True)


class AWD_LSTM(Module):
    "AWD-LSTM inspired by https://arxiv.org/abs/1708.02182"
    initrange=0.1

    def __init__(self, vocab_sz, emb_sz, n_hid, n_layers, pad_token=1, hidden_p=0.2, input_p=0.6, embed_p=0.1,
                 weight_p=0.5, bidir=False):
        store_attr('emb_sz,n_hid,n_layers,pad_token')
        self.bs = 1
        self.n_dir = 2 if bidir else 1
        self.encoder = nn.Embedding(vocab_sz, emb_sz, padding_idx=pad_token)
        self.encoder_dp = EmbeddingDropout(self.encoder, embed_p)
        self.rnns = nn.ModuleList([self._one_rnn(emb_sz if l == 0 else n_hid, (n_hid if l != n_layers - 1 else emb_sz)//self.n_dir,
                                                 bidir, weight_p, l) for l in range(n_layers)])
        self.encoder.weight.data.uniform_(-self.initrange, self.initrange)
        self.input_dp = RNNDropout(input_p)
        self.hidden_dps = nn.ModuleList([RNNDropout(hidden_p) for l in range(n_layers)])
        self.reset()

    def forward(self, inp, from_embeds=False):
        bs,sl = inp.shape[:2] if from_embeds else inp.shape
        if bs!=self.bs: self._change_hidden(bs)

        output = self.input_dp(inp if from_embeds else self.encoder_dp(inp))
        new_hidden = []
        for l, (rnn,hid_dp) in enumerate(zip(self.rnns, self.hidden_dps)):
            output, new_h = rnn(output, self.hidden[l])
            new_hidden.append(new_h)
            if l != self.n_layers - 1: output = hid_dp(output)
        self.hidden = to_detach(new_hidden, cpu=False, gather=False)
        return output

    def _change_hidden(self, bs):
        self.hidden = [self._change_one_hidden(l, bs) for l in range(self.n_layers)]
        self.bs = bs

    def _one_rnn(self, n_in, n_out, bidir, weight_p, l):
        "Return one of the inner rnn"
        rnn = nn.LSTM(n_in, n_out, 1, batch_first=True, bidirectional=bidir)
        return WeightDropout(rnn, weight_p)

    def _one_hidden(self, l):
        "Return one hidden state"
        nh = (self.n_hid if l != self.n_layers - 1 else self.emb_sz) // self.n_dir
        return (one_param(self).new_zeros(self.n_dir, self.bs, nh), one_param(self).new_zeros(self.n_dir, self.bs, nh))

    def _change_one_hidden(self, l, bs):
        if self.bs < bs:
            nh = (self.n_hid if l != self.n_layers - 1 else self.emb_sz) // self.n_dir
            return tuple(torch.cat([h, h.new_zeros(self.n_dir, bs-self.bs, nh)], dim=1) for h in self.hidden[l])
        if self.bs > bs: return (self.hidden[l][0][:,:bs].contiguous(), self.hidden[l][1][:,:bs].contiguous())
        return self.hidden[l]

    def reset(self):
        "Reset the hidden states"
        [r.reset() for r in self.rnns if hasattr(r, 'reset')]
        self.hidden = [self._one_hidden(l) for l in range(self.n_layers)]


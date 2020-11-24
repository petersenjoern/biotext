#%%

import sentencepiece as spm
from pathlib import Path
from collections import Counter, defaultdict
import torch
from torch import as_tensor,Tensor
import pandas as pd
import numpy as np
from numpy import array, ndarray

class SubWordTok():
    "SentencePiece tokenizer"
    def __init__(self, cache_dir='tmp', vocab_sz=None, max_vocab_sz=30000,
        char_coverage=0.99997, model_type='unigram') -> None:

        self.cache_dir = Path(cache_dir)
        self.vocab_sz, self.max_vocab_sz = vocab_sz, max_vocab_sz
        self.char_coverage = char_coverage
        self.model_type = model_type
        if (self.cache_dir/'spm.model').exists():
            self.tok = spm.SentencePieceProcessor()
            self.tok.Load(str(self.cache_dir/'spm.model'))


    def _get_vocab_sz(self, raw_text_path):
        "calc vocab sz and max vocab sz"
        cnt = Counter()
        with open(raw_text_path, 'r') as f:
            for line in f.readlines():
                cnt.update(line.split())
                if len(cnt)//4 > self.max_vocab_sz: return self.max_vocab_sz
        res = len(cnt)//4
        while res%8 != 0: res+=1
        return max(res,29)

    def train(self, raw_text_path):
        "Train a sentencepiece tokenizer on raw_text and save it"
        vocab_sz = self._get_vocab_sz(raw_text_path) if self.vocab_sz is None else self.vocab_sz
        spm.SentencePieceTrainer.Train(" ".join([
        f"--input={raw_text_path} --vocab_size={vocab_sz} --model_prefix={self.cache_dir/'spm'}",
        f"--character_coverage={self.char_coverage} --model_type={self.model_type}",
        "--pad_id=-1 --bos_id=-1 --eos_id=-1 --hard_vocab_limit=false"]))
        raw_text_path.unlink()
        return self.cache_dir/'spm.model'


    def setup(self, items):
        "In the setup a the train function is called with params"
        # to make the function generic, items is a list which parses to
        # an intermediate file (texts.out)
        if self.tok is None:
            raw_text_path = self.cache_dir/'texts.out'
            with open(raw_text_path, 'w') as f:
                for txt in items:
                    f.write(f'{txt}\n')
            sp_model = self.train(raw_text_path)
            self.tok = spm.SentencePieceProcessor()
            self.tok.Load(str(sp_model))

    def __call__(self, items):
        if self.tok is None: self.setup(items)
        for t in items: yield self.tok.EncodeAsPieces(t)
        

def make_vocab(count, min_freq=3, max_vocab=60000):
    "Create a vocab of `max_vocab` size from `Counter` `count` with items present more than `min_freq`"
    vocab = [o for o,c in count.most_common(max_vocab) if c >= min_freq]
    vocab = vocab[:max_vocab]
    return vocab + [f'xxfake' for i in range(0, 8-len(vocab)%8)]


def _array2tensor(x):
    "if ndarray return torch.from_numpy."
    if x.dtype==np.uint16: x = x.astype(np.float32)
    return torch.from_numpy(x)

def tensor(x, **kwargs):
    "Like `torch.as_tensor`, but handle lists too, and can pass multiple vector elements directly."
    res = (x if isinstance(x, Tensor)
           else torch.tensor(x, **kwargs) if isinstance(x, (tuple,list))
           else _array2tensor(x) if isinstance(x, ndarray)
           else as_tensor(x.values, **kwargs) if isinstance(x, (pd.Series, pd.DataFrame))
           else _array2tensor(array(x), **kwargs))
    if res.dtype is torch.float64: return res.float()
    return res

class Numericalize():
    "transform of tokenized texts to numericalized ids (tensors)"
    def __init__(self, vocab=None, min_freq=3, max_vocab=60000) -> None:
        self.vocab = vocab
        self.min_freq = min_freq
        self.max_vocab = max_vocab
        self.o2i = None if vocab is None else defaultdict(int, {v:k for k,v in enumerate(vocab)})

    def setup(self, dsets):
        "if vocab is not parsed, create the vocab and prepare o2i"
        if self.vocab is None:
            count = Counter(p for o in dsets for p in o)
            self.vocab = make_vocab(count, min_freq=self.min_freq, max_vocab=self.max_vocab)
            self.o2i = defaultdict(int, {v:k for k,v in enumerate(self.vocab) if v != 'xxfake'})
        # print(self.o2i)

    def encode(self, o): return tensor([self.o2i[o_] for o_ in o])
    def decode(self, o): return [self.vocab[o_] for o_ in o]

num = Numericalize(min_freq=3)
num.setup("This is a test setup, lets see what happens".split())
num.encode("This is a")
        



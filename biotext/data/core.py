#%%

import itertools
from typing import List, Generator
from fastai.data.core import TfmdDL
from fastcore.meta import delegates
import sentencepiece as spm
from pathlib import Path
from collections import Counter, defaultdict
import torch
from torch import as_tensor,Tensor
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from numpy import array, ndarray
import io
import pickle
import bz2
import gzip
import functools
from fastcore.foundation import L
import random
import sys


def make_vocab(count:Counter, min_freq=3, max_vocab=60000):
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

def open_file(fn, mode='r'):
    "Open a file, with optional compression if gz or bz2 suffix"
    if isinstance(fn, io.IOBase): return fn
    fn = Path(fn)
    if   fn.suffix=='.bz2': return bz2.BZ2File(fn, mode)
    elif fn.suffix=='.gz' : return gzip.GzipFile(fn, mode)
    else: return open(fn,mode)

def save_pickle(fn, o):
    "Save a pickle file, to a file name or opened file"
    with open_file(fn, 'wb') as f: pickle.dump(o, f)

def load_pickle(fn):
    "Load a pickle file from a file name or opened file"
    with open_file(fn, 'rb') as f: return pickle.load(f)


class SubWordTok():
    "SentencePiece tokenizer"
    def __init__(self, items:List, cache_dir='tmp', vocab_sz=None, max_vocab_sz=30000,
        char_coverage=0.99997, model_type='unigram') -> None:

        self.cache_dir = Path(cache_dir)
        self.vocab_sz, self.max_vocab_sz = vocab_sz, max_vocab_sz
        self.char_coverage = char_coverage
        self.model_type = model_type
        self._check_cache(items)

    def _check_cache(self, items):
        if (self.cache_dir/'spm.model').exists():
            self.tok = spm.SentencePieceProcessor()
            self.tok.Load(str(self.cache_dir/'spm.model'))
            print('reloaded tok file')
        else:
            self.setup(items)

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


    def setup(self, items: List, retrain=False):
        "In the setup a the train function is called with params"
        # to make the function generic, items is a list which parses to
        # an intermediate file (texts.out)
        if (self.tok is None) or retrain:
            raw_text_path = self.cache_dir/'texts.out'
            with open(raw_text_path, 'w') as f:
                for txt in items:
                    f.write(f'{txt}\n')
            sp_model = self.train(raw_text_path)
            self.tok = spm.SentencePieceProcessor()
            self.tok.Load(str(sp_model))

    def __call__(self, items: List):
        if self.tok is None: self.setup(items)
        for t in items: yield self.tok.EncodeAsPieces(t)
        


class Numericalize():
    "transform of tokenized texts to numericalized ids (tensors)"
    def __init__(self, cache_dir='tmp', dsets:List=None, vocab=None, min_freq=3, max_vocab=60000) -> None:
        
        self.cache_dir = Path(cache_dir)
        self.vocab = vocab
        self.min_freq = min_freq
        self.max_vocab = max_vocab
        self.o2i = None if vocab is None else defaultdict(int, {v:k for k,v in enumerate(vocab)})
        self._check_cache(dsets)

    def _check_cache(self, dsets):
        if (self.cache_dir/'num.pkl').exists():
            self.o2i = load_pickle(self.cache_dir/'num.pkl')
            print('reloaded num file')
        else:
            self.setup(dsets)

    def setup(self, dsets):
        "if vocab is not parsed, create the vocab and prepare o2i"
        if self.o2i is None:
            count = Counter(p for o in dsets for p in o)
            self.vocab = make_vocab(count, min_freq=self.min_freq, max_vocab=self.max_vocab)
            self.o2i = defaultdict(int, {v:k for k,v in enumerate(self.vocab) if v != 'xxfake'})
            save_pickle(self.cache_dir/'num.pkl', self.o2i)

    def encode(self, o): return tensor([self.o2i[o_] for o_ in o])
    def __call__(self, o): return tensor([self.o2i[o_] for o_ in o])
    def decode(self, o): return [self.vocab[o_] for o_ in o]

        
class Datasets(Dataset):
    "create a dataset"

    def __init__(self, items:List=None, tok=None, num=None):
        self.items = items
        self.tok = tok
        self.num = num
    
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        """
        tok __call__ requires a list('with a sentence').
        num __call__/encode requires ['with', 'a', 'sentence'].
        """
        sample = self.items[idx]
        item = [self.num(t) for t in self.tok([sample])][0]

        return item


def _maybe_first(o): return o[0] if isinstance(o, tuple) else o

def noop (x=None, *args, **kwargs):
    "Do nothing"
    return x

def is_listy(x):
    "`isinstance(x, (tuple,list,L,slice,Generator))`"
    return isinstance(x, (tuple,list,L,slice,Generator))

def ifnone(a, b):
    "`b` if `a` is None else `a`"
    return b if a is None else a

def round_multiple(x, mult, round_down=False):
    "Round `x` to nearest multiple of `mult`"
    def _f(x_): return (int if round_down else round)(x_/mult)*mult
    res = L(x).map(_f)
    return res if is_listy(x) else res[0]

def concat(*ls):
    "Concatenate tensors, arrays, lists, or tuples"
    if not len(ls): return []
    it = ls[0]
    if isinstance(it,torch.Tensor): res = torch.cat(ls)
    elif isinstance(it,ndarray): res = np.concatenate(ls)
    else:
        res = itertools.chain.from_iterable(map(L,ls))
        if isinstance(it,(tuple,list)): res = type(it)(res)
        else: res = L(res)
    return res

class ReindexCollection():
    "Reindexes collection `coll` with indices `idxs` and optional LRU cache of size `cache`"
    def __init__(self, coll, idxs=None, cache=None, tfm=noop):
        if idxs is None: idxs = L.range(coll)
        self.coll = coll
        self.idxs = idxs
        self.tfm = tfm
        if cache is not None: self._get = functools.lru_cache(maxsize=cache)(self._get)

    def _get(self, i): return self.tfm(self.coll[i])
    def __getitem__(self, i): return self._get(self.idxs[i])
    def __len__(self): return len(self.coll)
    # def reindex(self, idxs): self.idxs = idxs
    # def shuffle(self): random.shuffle(self.idxs)
    # def cache_clear(self): self._get.cache_clear()
    # def __getstate__(self): return {'coll': self.coll, 'idxs': self.idxs, 'cache': self.cache, 'tfm': self.tfm}
    # def __setstate__(self, s): self.coll,self.idxs,self.cache,self.tfm = s['coll'],s['idxs'],s['cache'],s['tfm']


class Chunks:
    "Slice and int indexing into a list of lists"
    def __init__(self, chunks, lens=None):
        self.chunks = chunks
        self.lens = L(map(len,self.chunks) if lens is None else lens)
        self.cumlens = np.cumsum(0+self.lens)
        self.totlen = self.cumlens[-1]

    def __getitem__(self,i):
        if isinstance(i,slice): return self.getslice(i)
        di,idx = self.doc_idx(i)
        return self.chunks[di][idx]

    def getslice(self, i):
        st_d,st_i = self.doc_idx(ifnone(i.start,0))
        en_d,en_i = self.doc_idx(ifnone(i.stop,self.totlen+1))
        res = [self.chunks[st_d][st_i:(en_i if st_d==en_d else sys.maxsize)]]
        for b in range(st_d+1,en_d): res.append(self.chunks[b])
        if st_d!=en_d and en_d<len(self.chunks): res.append(self.chunks[en_d][:en_i])
        return concat(*res)

    def doc_idx(self, i):
        if i<0: i=self.totlen+i # count from end
        docidx = np.searchsorted(self.cumlens, i+1)-1
        cl = self.cumlens[docidx]
        return docidx,i-cl

class LMDataLoader(TfmdDL):
    def __init__(self, dataset, cache=2, lens=None, bs=64, seq_len=72,
        num_workers=0, **kwargs):
        self.items = ReindexCollection(dataset, cache=cache, tfm=_maybe_first)
        if lens is None: lens = [len(o) for o in self.items]
        self.lens = lens
        self.seq_len = seq_len
        self.bs = bs

        corpus = round_multiple(sum(lens)-1, bs, round_down=True)
        self.bl = corpus//bs #bl stands for batch length
        self.n_batches = self.bl//(seq_len) + int(self.bl%seq_len!=0)
        self.last_len = self.bl - (self.n_batches-1)*seq_len
        self.n = self.n_batches*bs
        self.make_chunks()
        super().__init__(dataset=dataset, bs=bs, num_workers=num_workers, **kwargs)

    def make_chunks(self):
        self.chunks = Chunks(self.items, self.lens)
    
    def create_item(self, seq):
        if seq>=self.n: raise IndexError
        sl = self.last_len if seq//self.bs==self.n_batches-1 else self.seq_len
        st = (seq%self.bs)*self.bl + (seq//self.bs)*self.seq_len
        txt = self.chunks[st : st+sl+1]
        return tensor(txt[:-1]),txt[1:]


bs,sl = 4,3
ints = L([0,1,2,3,4],[5,6,7,8,9,10],[11,12,13,14,15,16,17,18],[19,20],[21,22]).map(tensor)
dl = LMDataLoader(ints, bs=bs, seq_len=sl)
for x,y in dl:
    print(f'This is x: {x}')
    print(f'This is y: {y}')

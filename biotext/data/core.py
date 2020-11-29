#%%

import itertools
from types import SimpleNamespace
from typing import List, Generator, Iterator, Sequence
from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter,_SingleProcessDataLoaderIter,_DatasetKind
_loaders = (_MultiProcessingDataLoaderIter,_SingleProcessDataLoaderIter)
import multiprocessing
from torch.utils.data import get_worker_info
from fastai.torch_core import default_device
from torch.utils.data._utils.collate import default_collate,default_convert
import typing
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
import math
import os
from contextlib import contextmanager



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

                try:
                    self.vocab = load_pickle(self.cache_dir/'vocab.pkl')
                except FileNotFoundError:
                    print('Cannot find vocab.pkl file')
        else:
            self.setup(dsets)

    def setup(self, dsets):
        "if vocab is not parsed, create the vocab and prepare o2i"
        if self.o2i is None:
            count = Counter(p for o in dsets for p in o)
            self.vocab = make_vocab(count, min_freq=self.min_freq, max_vocab=self.max_vocab)
            self.o2i = defaultdict(int, {v:k for k,v in enumerate(self.vocab) if v != 'xxfake'})
            save_pickle(self.cache_dir/'num.pkl', self.o2i)
            save_pickle(self.cache_dir/'vocab.pkl', self.vocab)

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

def noops(self, x=None, *args, **kwargs):
    "Do nothing (method)"
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


def chunked(it, chunk_sz=None, drop_last=False, n_chunks=None):
    "Return batches from iterator `it` of size `chunk_sz` (or return `n_chunks` total)"
    assert bool(chunk_sz) ^ bool(n_chunks)
    if n_chunks: chunk_sz = math.ceil(len(it)/n_chunks)
    if not isinstance(it, Iterator): it = iter(it)
    while True:
        res = list(itertools.islice(it, chunk_sz))
        if res and (len(res)==chunk_sz or not drop_last): yield res
        if len(res)<chunk_sz: return


class _InfMeta(type):
    @property
    def count(self): return itertools.count()
    @property
    def zeros(self): return itertools.cycle([0])
    @property
    def ones(self):  return itertools.cycle([1])
    @property
    def nones(self): return itertools.cycle([None])

# Cell
class Inf(metaclass=_InfMeta):
    "Infinite lists"
    pass

def set_num_threads(nt):
    "Get numpy (and others) to use `nt` threads"
    # try: import mkl; mkl.set_num_threads(nt)
    # except: pass
    try: import torch; torch.set_num_threads(nt)
    except: pass
    os.environ['IPC_ENABLE']='1'
    for o in ['OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
        os.environ[o] = str(nt)

def set_seed(s, reproducible=False):
    "Set random seed for `random`, `torch`, and `numpy` (where available)"
    try: torch.manual_seed(s)
    except NameError: pass
    try: torch.cuda.manual_seed_all(s)
    except NameError: pass
    try: np.random.seed(s%(2**32-1))
    except NameError: pass
    random.seed(s)
    if reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def _wif(worker_id):
    set_num_threads(1)
    info = get_worker_info()
    ds = info.dataset.d
    ds.num_workers,ds.offs = info.num_workers,info.id
    set_seed(info.seed)
    ds.wif()

def apply(func, x, *args, **kwargs):
    "Apply `func` recursively to `x`, passing on args"
    if is_listy(x): return type(x)([apply(func, o, *args, **kwargs) for o in x])
    if isinstance(x,dict):  return {k: apply(func, v, *args, **kwargs) for k,v in x.items()}
    res = func(x, *args, **kwargs)
    return res if x is None else res


defaults = SimpleNamespace()
def to_device(b, device=None):
    "Recursively put `b` on `device`."
    if defaults.use_cuda==False: device='cpu'
    elif device is None: device=default_device()
    def _inner(o): return o.to(device, non_blocking=True) if isinstance(o,Tensor) else o.to_device(device) if hasattr(o, "to_device") else o
    return apply(_inner, b)

class _FakeLoader:
    _IterableDataset_len_called,_auto_collation,collate_fn,drop_last = None,False,noops,False
    _index_sampler,generator,prefetch_factor  = Inf.count,None,2
    dataset_kind = _dataset_kind = _DatasetKind.Iterable
    def __init__(self, d, pin_memory, num_workers, timeout, persistent_workers):
        self.dataset,self.default,self.worker_init_fn = self,d,_wif
        self.d = d #This is the LMDataLoaderX(DataLoaderX) class
        self.pin_memory = pin_memory
        self.num_workers= num_workers
        self.timeout=timeout
        self.persistent_workers=persistent_workers

    def __iter__(self):
        print(f"3. iterate the yielded _loaders object, with create_batches")
        return iter(self.d.create_batches(self.d.sample()))

    @property
    def multiprocessing_context(self): return (None, multiprocessing)[self.num_workers>0]

    @contextmanager
    def no_multiproc(self):
        old_num_workers = self.num_workers
        try:
            self.num_workers = 0
            yield self.d
        finally: self.num_workers = old_num_workers

_collate_types = (ndarray, Tensor, typing.Mapping, str)

def fa_collate(t):
    "A replacement for PyTorch `default_collate` which maintains types and handles `Sequence`s"
    b = t[0]
    return (default_collate(t) if isinstance(b, _collate_types)
            else type(t[0])([fa_collate(s) for s in zip(*t)]) if isinstance(b, Sequence)
            else default_collate(t))

def fa_convert(t):
    "A replacement for PyTorch `default_convert` which maintains types and handles `Sequence`s"
    return (default_convert(t) if isinstance(t, _collate_types)
            else type(t)([fa_convert(s) for s in t]) if isinstance(t, Sequence)
            else default_convert(t))

#TODO: replace again with fastai DataLoader, this is only for inspection
class DataLoaderX:
    _noop_methods = 'wif before_iter after_item before_batch after_batch after_iter'.split()
    for o in _noop_methods: exec(f"def {o}(self, x=None, *args, **kwargs): return x")
    def __init__(self, dataset=None, bs=None, num_workers=0, pin_memory=False, timeout=0, batch_size=None,
    shuffle=False, drop_last=False, indexed=None, n=None, device=None, persistent_workers=False, **kwargs):
        
        if batch_size is not None: bs = batch_size # PyTorch compatibility
        assert not (bs is None and drop_last)
        if indexed is None: indexed = dataset is not None and hasattr(dataset,'__getitem__')
        if n is None:
            try: n = len(dataset)
            except TypeError: pass
        self.dataset = dataset
        self.bs = bs
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.indexed = indexed
        self.n = n
        if num_workers is None: num_workers = 1
        self.num_workers = num_workers
        self.device = device
        self.rng,self.num_workers,self.offs = random.Random(random.randint(0,2**32-1)),1,0
        self.fake_l = _FakeLoader(self, pin_memory, num_workers, timeout, persistent_workers=persistent_workers)

    @property
    def prebatched(self): return self.bs is None
    # def randomize(self): self.rng = random.Random(self.rng.randint(0,2**32-1))
    def chunkify(self, b):
        print(f"chunkify: {[b for b in chunked(b, self.bs, self.drop_last)]}")
        return b if self.prebatched else chunked(b, self.bs, self.drop_last)
    
    def create_batches(self, samps):
        print(f"5. create_batches: make iter(dataset)")
        self.it = iter(self.dataset) if self.dataset is not None else None
        print(f"5. iter(dataset): {[t for t in self.it]}")
        print(f"5. samples: {[s for s in samps]}")
        res = filter(lambda o:o is not None, map(self.do_item, samps))
        print(f"print res: {[r for r in res]}")
        yield from map(self.do_batch, self.chunkify(res))

    def do_item(self, s):
        return self.after_item(self.create_item(s))

    def get_idxs(self):
        print(f"1. create idxs from n")
        print(f"1. n is {self.n}")
        idxs = Inf.count if self.indexed else Inf.nones
        if self.n is not None: idxs = list(itertools.islice(idxs, self.n))
        # if self.shuffle: idxs = self.shuffle_fn(idxs)
        print(f"1. idxs are: {idxs}")
        return idxs

    def sample(self):
        print(f"4. prepare sample")
        print(f"4. {[b for i,b in enumerate(self.__idxs) if i//(self.bs or 1)%self.num_workers==self.offs]}")
        return (b for i,b in enumerate(self.__idxs) if i//(self.bs or 1)%self.num_workers==self.offs)

    def do_batch(self, b):
        print(f"do_batch: {self.create_batch(self.before_batch(b)), b}")
        return self.retain(self.create_batch(self.before_batch(b)), b)
    def retain(self, res, b):  return res
    def create_batch(self, b): return (fa_collate,fa_convert)[self.prebatched](b)

    def __iter__(self):
        # self.randomize()
        self.__idxs=self.get_idxs() # called in context of main process (not workers/subprocesses)
        print(f"2. create _loaders: {_loaders[self.fake_l.num_workers==0](self.fake_l)} and yield them")
        for b in _loaders[self.fake_l.num_workers==0](self.fake_l):
            if self.device is not None: b = to_device(b, self.device)
            yield self.after_batch(b)
        # self.after_iter()
        # if hasattr(self, 'it'): del(self.it)

class LMDataLoaderX(DataLoaderX):
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
        print(f"6. create_item: {tensor(txt[:-1]),txt[1:]}")
        return tensor(txt[:-1]),txt[1:]


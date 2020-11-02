"""
Module for Tokenizer

SentencePieceTokenizer wrapped in fastai Tokenizer()


"""
## TODO: CLEANUP ONLY NEEDED OBJECTS

#%%

import os
import pandas as pd

from typing import List
from pathlib import Path


from sentencepiece import SentencePieceTrainer, SentencePieceProcessor
from collections import Counter
from types import SimpleNamespace

from fastai.imports import pv
from fastai.data.core import TfmdDL, DataLoaders
from fastai.data.transforms import CategoryMap, Category
from fastai.torch_core import TensorCategory

from fastcore.foundation import L, first, GetAttr, mask2idxs, is_indexer
from fastcore.utils import parallel_gen, compose, store_attr, maps, merge, add_props, is_listy
from fastcore.transform import Transform, mk_transform
from fastcore.meta import delegates

from fastprogress import progress_bar

#%%
## Set Variables
UNK, PAD, BOS, EOS, FLD, TK_REP, TK_WREP, TK_UP, TK_MAJ = "xxunk xxpad xxbos xxeos xxfld xxrep xxwrep xxup xxmaj".split()
eu_langs = ["bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "ga", "hr", "hu",
            "it","lt","lv","mt","nl","pl","pt","ro","sk","sl","sv"] # all European langs

#%% 
def num_cpus():
    "Get number of cpus"
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError: 
        return os.cpu_count()

def replace_space(t):
    "Replace embedded spaces in a token with unicode line char to allow for split/join"
    return t.replace(' ', '▁')


def lowercase(t, add_bos=True, add_eos=False):
    "Converts `t` to lowercase"
    return (f'{BOS} ' if add_bos else '') + t.lower().strip() + (f' {EOS}' if add_eos else '')

## Set defaults that are callable later on
defaults = SimpleNamespace()
defaults.text_spec_tok = [UNK, PAD, BOS, EOS, FLD, TK_REP, TK_WREP, TK_UP, TK_MAJ]
defaults.cpus = num_cpus()
defaults.text_postproc_rules = [replace_space]
defaults.text_proc_rules = [lowercase]

#%%

def noop (x=None, *args, **kwargs):
    "Do nothing"
    return x


def compose_tfmsx(x, tfms, is_enc=True, reverse=False, **kwargs):
    "Apply all `func_nm` attribute of `tfms` on `x`, maybe in `reverse` order"
    if reverse: tfms = reversed(tfms)
    for f in tfms:
        if not is_enc: f = f.decode
        x = f(x, **kwargs)
    return x

class PipelineX:
    "A pipeline of composed (for encode/decode) transforms, setup with types"
    def __init__(self, funcs=None, split_idx=None):
        self.split_idx,self.default = split_idx,None

        if funcs is None: funcs = [] # funcs have to be a list

        else:
            if isinstance(funcs, Transform): # test if funcs is a single Transform obj
                funcs = [funcs] # if so, return list with Transform obj
            self.fs = L(ifnone(funcs,[noop])).map(mk_transform).sorted(key='order') #instantiate every Transformer

    def __call__(self, o): return compose_tfmsx(o, tfms=self.fs, split_idx=self.split_idx)
    def __repr__(self): return f"Pipeline: {' -> '.join([f.name for f in self.fs if f.name != 'noop'])}"

    def decode  (self, o, full=True):
        if full: return compose_tfmsx(o, tfms=self.fs, is_enc=False, reverse=True, split_idx=self.split_idx)
        #Not full means we decode up to the point the item knows how to show itself.
        for f in reversed(self.fs):
            if self._is_showable(o): return o
            o = f.decode(o, split_idx=self.split_idx)
        return o

    def setup(self, items=None, train_setup=False):
        tfms = self.fs[:]
        self.fs.clear()
        for t in tfms:
            self.add(t,items, train_setup)
            print(t)
    
    def add(self,t, items=None, train_setup=False):
        t.setup(items, train_setup)
        self.fs.append(t)
        print(self.fs)

#%%

class DisplayedTransform(Transform):
    "A transform with a `__repr__` that shows its attrs"

    @property
    def name(self): return f"{super().name} -- {getattr(self,'__stored_args__',{})}"


class CategorizeX(DisplayedTransform):
    "Reversible transform of category string to `vocab` id"
    # loss_func,order=CrossEntropyLossFlat(),1
    def __init__(self, vocab=None, sort=True, add_na=False):
        if vocab is not None: vocab = CategoryMap(vocab, sort=sort, add_na=add_na)
        store_attr()

    def setups(self, dsets):
        if self.vocab is None and dsets is not None: self.vocab = CategoryMap(dsets, sort=self.sort, add_na=self.add_na)
        self.c = len(self.vocab)

    def encodes(self, o):
        try:
            return TensorCategory(self.vocab.o2i[o])
        except KeyError as e:
            raise KeyError(f"Label '{o}' was not included in the training dataset") from e
    def decodes(self, o): return Category      (self.vocab    [o])

#%%
class FilteredBase:
    "Base class for lists with subsets"
    _dl_type,_dbunch_type = TfmdDL,DataLoaders
    def __init__(self, *args, dl_type=None, **kwargs):
        if dl_type is not None: self._dl_type = dl_type
        self.dataloaders = delegates(self._dl_type.__init__)(self.dataloaders)
        super().__init__(*args, **kwargs)

    @property
    def n_subsets(self): return len(self.splits)
    def _new(self, items, **kwargs): return super()._new(items, splits=self.splits, **kwargs)
    def subset(self): raise NotImplemented

    def dataloaders(self, bs=64, val_bs=None, shuffle_train=True, n=None, path='.', dl_type=None, dl_kwargs=None,
                    device=None, **kwargs):
        if device is None: device=default_device()
        if dl_kwargs is None: dl_kwargs = [{}] * self.n_subsets
        if dl_type is None: dl_type = self._dl_type
        drop_last = kwargs.pop('drop_last', shuffle_train)
        dl = dl_type(self.subset(0), bs=bs, shuffle=shuffle_train, drop_last=drop_last, n=n, device=device,
                     **merge(kwargs, dl_kwargs[0]))
        dls = [dl] + [dl.new(self.subset(i), bs=(bs if val_bs is None else val_bs), shuffle=False, drop_last=False,
                             n=None, **dl_kwargs[i]) for i in range(1, self.n_subsets)]
        return self._dbunch_type(*dls, path=path, device=device)

FilteredBase.train,FilteredBase.valid = add_props(lambda i,x: x.subset(i))



class TfmdListsX(FilteredBase, L, GetAttr):
    "A `Pipeline` of `tfms` applied to a collection of `items`"
    _default='tfms'
    def __init__(self, items, tfms, use_list=None, do_setup=True, split_idx=None, train_setup=True,
                 splits=None, types=None, verbose=False, dl_type=None):
        super().__init__(items, use_list=use_list)
        if dl_type is not None: self._dl_type = dl_type

        #potentially unused
        self.splits = L([slice(None),[]] if splits is None else splits).map(mask2idxs)
        if isinstance(tfms,TfmdListsX): tfms = tfms.tfms
        if isinstance(tfms,PipelineX): do_setup=False


        # This is relevant, equivalent to PipelineX
        self.tfms = PipelineX(tfms, split_idx=split_idx)

        store_attr('types,split_idx')
        if do_setup:
            pv(f"Setting up {self.tfms}", verbose)
            self.setup(train_setup=train_setup)

    def _new(self, items, split_idx=None, **kwargs):
        split_idx = ifnone(split_idx,self.split_idx)
        return super()._new(items, tfms=self.tfms, do_setup=False, types=self.types, split_idx=split_idx, **kwargs)
    def subset(self, i): return self._new(self._get(self.splits[i]), split_idx=i)
    def _after_item(self, o): return self.tfms(o)
    def __repr__(self): return f"{self.__class__.__name__}: {self.items}\ntfms - {self.tfms.fs}"
    def __iter__(self): return (self[i] for i in range(len(self)))
    def show(self, o, **kwargs): return self.tfms.show(o, **kwargs)
    def decode(self, o, **kwargs): return self.tfms.decode(o, **kwargs)
    def __call__(self, o, **kwargs): return self.tfms.__call__(o, **kwargs)
    def overlapping_splits(self): return L(Counter(self.splits.concat()).values()).filter(gt(1))
    def new_empty(self): return self._new([])

    def setup(self, train_setup=True):
        self.tfms.setup(self, train_setup)
        if len(self) != 0:
            x = super().__getitem__(0) if self.splits is None else super().__getitem__(self.splits[0])[0]
            self.types = []
            for f in self.tfms.fs:
                self.types.append(getattr(f, 'input_types', type(x)))
                x = f(x)
            self.types.append(type(x))
        types = L(t if is_listy(t) else [t] for t in self.types).concat().unique()
        self.pretty_types = '\n'.join([f'  - {t}' for t in types])

    def __getitem__(self, idx):
        res = super().__getitem__(idx)
        if self._after_item is None: return res
        return self._after_item(res) if is_indexer(idx) else res.map(self._after_item)


#%%

def ifnone(a, b):
    "`b` if `a` is None else `a`"
    return b if a is None else a


def parallel_tokenize(items, tok=None, rules=None, n_workers=defaults.cpus, **kwargs):
    "Calls optional `setup` on `tok` before launching `TokenizeWithRules` using `parallel_gen"
    if tok is None: tok = SubWordTok()
    if hasattr(tok, 'setup'): tok.setup(items, rules)
    return parallel_gen(TokenizeWithRules, items, tok=tok, rules=rules, n_workers=n_workers, **kwargs)


def _join_texts(df, mark_fields=False):
    "Join texts in row `idx` of `df`, marking each field with `FLD` if `mark_fields=True`"
    text_col = (f'{FLD} {1} ' if mark_fields else '' ) + df.iloc[:,0].astype(str)
    for i in range(1,len(df.columns)):
        text_col += (f' {FLD} {i+1} ' if mark_fields else ' ') + df.iloc[:,i].astype(str)
    return text_col.values

def tokenize_df(df, text_cols, mark_fields=None, n_workers=defaults.cpus, rules: List=None, tok=None, res_col_name="text"):
    "Tokenize text in df[text_col]"
    text_cols = [df.columns[c] if isinstance(c, int) else c for c in L(text_cols)]
    #mark_fields defaults to False if there is one column of texts, True if there are multiple
    if mark_fields is None: mark_fields = len(text_cols)>1
    rules = L(ifnone(rules, defaults.text_proc_rules.copy()))
    texts = _join_texts(df[text_cols], mark_fields=mark_fields)
    outputs = L(parallel_tokenize(texts, tok, rules, n_workers=n_workers)
            ).sorted().itemgot(1)

    other_cols = df.columns[~df.columns.isin(text_cols)]
    res = df[other_cols].copy()
    res[res_col_name] = outputs
    res[f'{res_col_name}_length'] = [len(o) for o in outputs]
    return res,Counter(outputs.concat())


class Tokenizer(Transform):
    "Provides a consistent `Transform` interface to tokenizers operating on `DataFrame`s and folders"
    input_types = (str, list, L, tuple, Path)
    def __init__(self, tok, rules=None, counter=None, lengths=None, mode=None, sep=' '):
        if isinstance(tok,type): tok=tok()
        store_attr('tok,counter,lengths,mode,sep')
        self.rules = defaults.text_proc_rules if rules is None else rules

    @classmethod
    @delegates(tokenize_df, keep=True)
    def from_df(cls, text_cols, tok=None, rules=None, sep=' ', **kwargs):
        if tok is None: tok = SubWordTok()
        res = cls(tok, rules=rules, mode='df')
        res.kwargs,res.train_setup = merge({'tok': tok}, kwargs),False
        res.text_cols,res.sep = text_cols,sep
        return res

    def setups(self, dsets):
        if not self.mode == 'df' or not isinstance(dsets.items, pd.DataFrame): return
        dsets.items,count = tokenize_df(dsets.items, self.text_cols, rules=self.rules, **self.kwargs)
        if self.counter is None: self.counter = count
        return dsets

    def encodes(self, o:Path):
        if self.mode=='folder' and str(o).startswith(str(self.path)):
            tok = self.output_dir/o.relative_to(self.path)
            return L(tok.read_text(encoding="utf-8").split(' '))
        else: return self._tokenize1(o.read_text())

    def encodes(self, o:str): return self._tokenize1(o)
    def _tokenize1(self, o): return first(self.tok([compose(*self.rules)(o)]))

    def get_lengths(self, items):
        if self.lengths is None: return None
        if self.mode == 'df':
            if isinstance(items, pd.DataFrame) and 'text_lengths' in items.columns: return items['text_length'].values
        if self.mode == 'folder':
            try:
                res = [self.lengths[str(Path(i).relative_to(self.path))] for i in items]
                if len(res) == len(items): return res
            except: return None

    def decodes(self, o): return TitledStr(self.sep.join(o))


class TokenizeWithRules:
    "A wrapper around `tok` which applies `rules`, then tokenizes, then applies `post_rules`"
    def __init__(self, tok, rules=None, post_rules=None):
        self.rules = L(ifnone(rules, defaults.text_proc_rules))
        self.post_f = compose(*L(ifnone(post_rules, defaults.text_postproc_rules)))
        self.tok = tok

    def __call__(self, batch):
        return (L(o).map(self.post_f) for o in self.tok(maps(*self.rules, batch)))


class SpacyTokenizer():
    "Spacy tokenizer for `lang`"
    def __init__(self, lang='en', special_toks=None, buf_sz=5000):
        self.special_toks = ifnone(special_toks, defaults.text_spec_tok)
        nlp = spacy.blank(lang, disable=["parser", "tagger", "ner"])
        for w in self.special_toks: nlp.tokenizer.add_special_case(w, [{ORTH: w}])
        self.pipe,self.buf_sz = nlp.pipe,buf_sz

    def __call__(self, items):
        return (L(doc).attrgot('text') for doc in self.pipe(map(str,items), batch_size=self.buf_sz))

WordTokenizer = SpacyTokenizer


class SubWordTok():
    "SentencePiece tokenizer for `lang`"
    def __init__(self, lang='en', special_toks=None, sp_model=None, vocab_sz=None, max_vocab_sz=30000,
                 model_type='unigram', char_coverage=None, cache_dir='tmp'):

        self.sp_model,self.cache_dir = sp_model,Path(cache_dir)
        self.vocab_sz,self.max_vocab_sz,self.model_type = vocab_sz,max_vocab_sz,model_type
        self.char_coverage = ifnone(char_coverage, 0.99999 if lang in eu_langs else 0.9998)
        self.special_toks = ifnone(special_toks, defaults.text_spec_tok)
        if sp_model is None: self.tok = None
        else:
            self.tok = SentencePieceProcessor()
            self.tok.Load(str(sp_model))
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_vocab_sz(self, raw_text_path):
        cnt = Counter()
        with open(raw_text_path, 'r') as f:
            for line in f.readlines():
                cnt.update(line.split())
                if len(cnt)//4 > self.max_vocab_sz: return self.max_vocab_sz
        res = len(cnt)//4
        while res%8 != 0: res+=1
        return max(res,29)

    def train(self, raw_text_path):
        "Train a sentencepiece tokenizer on `texts` and save it in `path/tmp_dir`"
        vocab_sz = self._get_vocab_sz(raw_text_path) if self.vocab_sz is None else self.vocab_sz
        spec_tokens = ['\u2581'+s for s in self.special_toks]
        SentencePieceTrainer.Train(" ".join([
            f"--input={raw_text_path} --vocab_size={vocab_sz} --model_prefix={self.cache_dir/'spm'}",
            f"--character_coverage={self.char_coverage} --model_type={self.model_type}",
            f"--unk_id={len(spec_tokens)} --pad_id=-1 --bos_id=-1 --eos_id=-1 --minloglevel=2",
            f"--user_defined_symbols={','.join(spec_tokens)} --hard_vocab_limit=false"]))
        raw_text_path.unlink()
        return self.cache_dir/'spm.model'

    def setup(self, items, rules=None):
        if rules is None: rules = []
        if self.tok is not None: return {'sp_model': self.sp_model}
        raw_text_path = self.cache_dir/'texts.out'
        with open(raw_text_path, 'w') as f:
            for t in progress_bar(maps(*rules, items), total=len(items), leave=False):
                f.write(f'{t}\n')
        sp_model = self.train(raw_text_path)
        self.tok = SentencePieceProcessor()
        self.tok.Load(str(sp_model))
        return {'sp_model': sp_model}

    def __call__(self, items):
        if self.tok is None: self.setup(items)
        for t in items: yield self.tok.EncodeAsPieces(t)
"""
Module for Tokenizer

BaseTokenizer
SubwordTokenizer


"""
## TODO: parallel tokenize to speedup 

import os
from typing import List
from fastcore.foundation import L, first
from fastcore.utils import parallel_gen, compose, store_attr, maps, merge
from fastcore.transform import Transform
from fastcore.meta import delegates
from fastprogress import progress_bar
from pathlib import Path
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor
from collections import Counter
from types import SimpleNamespace
import pandas as pd


defaults = SimpleNamespace()


UNK, PAD, BOS, EOS, FLD, TK_REP, TK_WREP, TK_UP, TK_MAJ = "xxunk xxpad xxbos xxeos xxfld xxrep xxwrep xxup xxmaj".split()
defaults.text_spec_tok = [UNK, PAD, BOS, EOS, FLD, TK_REP, TK_WREP, TK_UP, TK_MAJ]

eu_langs = ["bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "ga", "hr", "hu",
            "it","lt","lv","mt","nl","pl","pt","ro","sk","sl","sv"] # all European langs

def num_cpus():
    "Get number of cpus"
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError: 
        return os.cpu_count()



defaults.cpus = num_cpus()

def replace_space(t):
    "Replace embedded spaces in a token with unicode line char to allow for split/join"
    return t.replace(' ', '▁')

defaults.text_postproc_rules = [replace_space]


def lowercase(t, add_bos=True, add_eos=False):
    "Converts `t` to lowercase"
    return (f'{BOS} ' if add_bos else '') + t.lower().strip() + (f' {EOS}' if add_eos else '')

defaults.text_proc_rules = [lowercase]

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

class BaseTokenizer():
    "A Tokenzier that splits on spaces"
    def __init__(self, split_char:str=' ', **kwargs): self.split_char = split_char
    def __call__(self, items: List): return (t.split(self.split_char) for t in items)

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
        if tok is None: tok = SentencePieceTokenizer()
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


def ifnone(a, b):
    "`b` if `a` is None else `a`"
    return b if a is None else a


def parallel_tokenize(items, tok=None, rules=None, n_workers=defaults.cpus, **kwargs):
    "Calls optional `setup` on `tok` before launching `TokenizeWithRules` using `parallel_gen"
    if tok is None: tok = SentencePieceTokenizer()
    if hasattr(tok, 'setup'): tok.setup(items, rules)
    return parallel_gen(TokenizeWithRules, items, tok=tok, rules=rules, n_workers=n_workers, **kwargs)





class SentencePieceTokenizer():
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
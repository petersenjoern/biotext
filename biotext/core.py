"""
Module for Tokenizer

BaseTokenizer
SubwordTokenizer


"""
## TODO: parallel tokenize to speedup 

import os
from typing import List
from fastcore.foundation import L
from fastcore.utils import parallel_gen, compose
from fastprogress import progress_bar
from pathlib import Path
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor
from collections import Counter
from types import SimpleNamespace


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


class BaseTokenizer():
    "A Tokenzier that splits on spaces"
    def __init__(self, split_char:str=' ', **kwargs): self.split_char = split_char
    def __call__(self, items: List): return (t.split(self.split_char) for t in items)


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


def tokenize_df(df, text_col:str, n_workers=defaults.cpus, rules: List=None, tok=None, res_col_name="text"):
    "Tokenize text in df[text_col]"
    # rules = L(ifnone(rules, defaults.text_proc_rules.copy()))
    text = df[text_col].values
    outputs = L(parallel_tokenize(text, tok, rules, n_workers=n_workers)
            ).sorted().itemgot(1)
    return Counter(outputs.concat())



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
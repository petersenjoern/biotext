#%%

import sentencepiece as spm
from pathlib import Path
from collections import Counter


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
        

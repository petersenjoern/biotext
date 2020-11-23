#%%

import sentencepiece as spm
from pathlib import Path
from collections import Counter




class SubWordTok():
    "SentencePiece tokenizer"
    def __init__(self, cache_dir='tmp', vocab_sz=None, max_vocab_sz=30000) -> None:
        self.cache_dir = Path(cache_dir)
        self.vocab_sz, self.max_vocab_sz = vocab_sz, max_vocab_sz

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
        "Train a sentencepiece tokenizer on raw_text and save it"
        vocab_sz = self._get_vocab_sz(raw_text_path) if self.vocab_sz is None else self.vocab_sz
        
        spm.SentencePieceTrainer.Train(" ".join([
        f"--input={raw_text_path} --vocab_size={vocab_sz} --model_prefix={self.cache_dir/'spm'}"]))

        self.tok = spm.SentencePieceProcessor()
        self.tok.Load(str(self.cache_dir/'spm.model'))
        return self.tok.EncodeAsPieces("test this out")
        


cache_dir = Path.cwd()
raw_text_path = cache_dir/'botchan.txt'

tok = SubWordTok(cache_dir)
tok.train(raw_text_path)


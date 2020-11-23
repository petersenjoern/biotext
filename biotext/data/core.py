#%%

import sentencepiece as spm
from pathlib import Path
import io


class SubWordTok():
    "SentencePiece tokenizer"
    def __init__(self) -> None:
        pass

    def train(self, raw_text_path):
        "Train a sentencepiece tokenizer on raw_text and save it"
        model = io.BytesIO()
        spm.SentencePieceTrainer.train(
            input=raw_text_path,
            model_prefix='m',
            vocab_size=1000,
            user_defined_symbols=['foo', 'bar'],
            model_writer=model)

        # path_out = raw_text_path.parent[0]/'spm.model'
        # with open(path_out, 'wb') as f:
        #     f.write(model.getvalue())
        
        sp = spm.SentencePieceProcessor(model_proto=model.getvalue())
        print(sp.encode('this is test'))
        


raw_text_path = Path.cwd()/'botchan.txt'

tok = SubWordTok()
tok.train(raw_text_path)


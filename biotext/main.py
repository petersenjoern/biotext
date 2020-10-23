"""
Module to collect data and model
"""

import os
from data.external import path
import pandas as pd
from core import tokenize_df, SentencePieceTokenizer, Tokenizer
from fastcore.foundation import L, first
from fastai.text.data import TextDataLoaders

# import torch
# from torch.nn import functional as F
# from torch.utils.data import DataLoader
# import pytorch_lightning as pl



if __name__ == "__main__":
    
    # Path to the data
    path_wikitext = path(fname="wikitext-2", c_key="data")
    path_wikitext_train = path_wikitext/'train.csv'
    path_wikitext_valid = path_wikitext/'test.csv'

    # Load data
    df_train = pd.read_csv(path_wikitext_train, header=None)
    df_valid = pd.read_csv(path_wikitext_valid, header=None)
    df_all = pd.concat([df_train, df_valid])
    df_all.columns = ["text"]
    print(df_all.head())

    # Apply SentencePieceTokenizer
    # txt = L(df_all['text'].tolist())
    # sp = SentencePieceTokenizer(vocab_sz=1000)
    # sp.setup(txt)
    # print(next(iter(sp(txt[:2]))))

    # Function to apply Tokenizer to df and get tokenizer results summary
    # out,cnt=tokenize_df(df=df_all, text_cols="text", tok=SentencePieceTokenizer)
    # print(out)
    # print(cnt)

    # Wrapper around Tokenizer (easy to use e.g. from_df)
    # toke=Tokenizer.from_df(df=df_all, text_cols="text", tok=SentencePieceTokenizer())
    # toke.tok.setup(L(toke.kwargs['df']['text'].tolist()))
    # print(first(toke.tok(L(toke.kwargs['df']['text'].tolist()[:2]))))
    # print(toke.__dict__)

    # Tokenize, Numericalize, Split, Shuffle, Offset by 1 (lm),Batch
    dl = TextDataLoaders.from_df(
        df=df_all,
        text_col="text",
        valid_pct=0.1,
        y_block=None,
        is_lm=True,
        tok=SentencePieceTokenizer(vocab_sz=2000),
        bs=32,
        seq_len=72,
        shuffle_train=True,
        verbose=True)
    print(dl.show_batch(max_n=3))

    # TODO:
    # Tokenizer in core.py
    # DataLoader in data/core.py (batchify data, prepare for langauge model)
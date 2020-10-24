"""
Module to collect data and model
"""

import os
from data.external import Config, path
import pandas as pd
from fastcore.foundation import L, first
from fastai.text.data import TextDataLoaders
from utils import *

# import torch
# from torch.nn import functional as F
# from torch.utils.data import DataLoader
# import pytorch_lightning as pl

LOADER_RETRAIN = False
FNAME = "wikitext-2"
VOCAB_SZ= 2000
BS = 32
SEQ_LEN = 72
VALID_PCT = 0.1

if __name__ == "__main__":

    # Intiate Configuration (dir's)
    Config()

    # Path to the models
    path_models = path(FNAME, c_key="model")
    path_models_dl = path_models/'TextDataLoaders.pkl'
    
    # Path to the data
    path_wikitext = path(FNAME, c_key="data")
    path_wikitext_train = path_wikitext/'train.csv'
    path_wikitext_valid = path_wikitext/'test.csv'

    # Load data
    df_train = pd.read_csv(path_wikitext_train, header=None)
    df_valid = pd.read_csv(path_wikitext_valid, header=None)
    df_all = pd.concat([df_train, df_valid])
    df_all.columns = ["text"]


    # Tokenize, Numericalize, Split, Shuffle, Offset by 1 (lm),Batch
    if LOADER_RETRAIN:
        tok = SentencePieceTokenizer(vocab_sz=VOCAB_SZ)
        dls = TextDataLoaders.from_df(
            df=df_all,
            text_col="text",
            valid_pct=VALID_PCT,
            y_block=None,
            is_lm=True,
            tok=tok,
            bs=BS,
            seq_len=SEQ_LEN,
            shuffle_train=True,
            verbose=True)
        pickle_save(dl, path_models_dl)
    else:
        dls = pickle_load(path_models_dl)

    TextDataLoadersInspector(dls)
    dls.show_batch(max_n=3)





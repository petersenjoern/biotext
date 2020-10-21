"""
Module to collect data and model
"""

import os
from data.external import path
import pandas as pd
from core import tokenize_df, SentencePieceTokenizer

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

    x=tokenize_df(df=df_all, text_col="text", tok=SentencePieceTokenizer)
    print(x)

    # TODO:
    # Tokenizer in core.py
    # DataLoader in data/core.py (batchify data, prepare for langauge model)
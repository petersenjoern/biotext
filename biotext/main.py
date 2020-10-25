"""
Module to collect data and model
"""

import os
import torch
import pandas as pd

from data.external import Config, path
from core import SubWordTok
from utils import TextDataLoadersInspector, pickle_save, pickle_load
from model import awd_lstm_lm_config

from fastai.text.all import *
# from fastai.text.data import TextDataLoaders
# from fastai.text.learner import language_model_learner
# from fastai.text.models.awdlstm import AWD_LSTM
from fastai.metrics import Perplexity, accuracy
from fastai.data.external import URLs
from fastai.callback import *



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
        # tok = SubWordTok(vocab_sz=VOCAB_SZ)
        tok = SubWordTok()
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
            verbose=True,
            device=torch.device('cuda') #somehow not recognised
            )
        pickle_save(dls, path_models_dl)
        LOADER_RETRAIN = False
    else:
        dls = pickle_load(path_models_dl)


    TextDataLoadersInspector(dls)
    dls.show_batch(max_n=3)


    # Train Model
    # Init
    tensorboard=URLs.LOCAL_PATH/'tmp'/'runs'/'init'
    cbs=TensorBoardCallback(tensorboard, trace_model=False) # Trace has to be false, because of mixed precision (FP16)
    learn = language_model_learner(
        dls=dls,
        arch=AWD_LSTM,
        config=awd_lstm_lm_config,
        pretrained=True,
        drop_mult=0.3,
        metrics=[accuracy, Perplexity()]
        ).to_fp16()
    print(learn.model)
    # print(learn.lr_find())
    # learn.fit_one_cycle(1, 6e-3, moms=(0.8,0.7,0.8), cbs=cbs)
    # learn.save("awd_lstm_1_epoch")

    # Continue in all layers
    tensorboard=URLs.LOCAL_PATH/'tmp'/'runs'/'unfreeze'
    cbs=TensorBoardCallback(tensorboard, trace_model=False) # Trace has to be false, because of mixed precision (FP16)
    learn = learn.load("awd_lstm_1_epoch")
    print(learn.lr_find())
    # learn.unfreeze()
    # learn.fit_one_cycle(10, 2e-3, moms=(0.8,0.7,0.8), cbs=cbs)
    # learn.save("awd_lstm_fine_tune")
    # learn.save_encoder("awd_lstm_fine_tune")



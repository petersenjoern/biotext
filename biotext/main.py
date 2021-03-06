"""
Module to collect data and model
"""

import os
import torch
import pandas as pd

from data.external import Config, path
from core import SubWordTok, WordTokenizer
from utils import TextDataLoadersInspector, pickle_save, pickle_load
from model import awd_lstm_lm_config

from fastai.text.all import *
from fastai.metrics import Perplexity, accuracy
from fastai.data.external import URLs
from fastai.callback.tensorboard import *


MODEL_NAME = "BIO_AWD_LSTM"
RETRAIN_DATA_LOADER = True

FIT_LM_1_EPOCH = False
LM_LR_FIND_1_EPOCH = False

FIT_LM_FINE_TUNE = False
LM_LR_FIND_FINE_TUNE = False

FNAME = "ct-gov"
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
    path_data = path(FNAME, c_key="data")
    path_data_lm = path_data/'clinical_trial_descriptions_sub.txt'
    
    # Load lm data
    df = pd.read_csv(path_data_lm, sep="\t", header=None)
    df.columns = ["text"]


    # # Tokenize, Numericalize, Split, Shuffle, Offset by 1 (lm), Batch all in parallel (cpus)
    if RETRAIN_DATA_LOADER:
        tok = WordTokenizer()
        dls = TextDataLoaders.from_df(
            df=df,
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
    else:
        dls = pickle_load(path_models_dl)


    TextDataLoadersInspector(dls)
    dls.show_batch(max_n=3)


    # Train Model
    # Init
    learn = language_model_learner(
        dls=dls,
        arch=AWD_LSTM,
        config=awd_lstm_lm_config,
        pretrained=True,
        drop_mult=0.3,
        metrics=[accuracy, Perplexity()]
        ).to_fp16()
    print(learn.model)

    if FIT_LM_1_EPOCH:
        if LM_LR_FIND_1_EPOCH:
            lr_min, lr_steep = learn.lr_find()

        tensorboard=URLs.LOCAL_PATH/'tmp'/'runs'/f"{MODEL_NAME}_1_EPOCH"
        cbs=TensorBoardCallback(tensorboard, trace_model=False) # Trace has to be false, because of mixed precision (FP16)
        
        if lr_min:
            print(f"LM 1 Epoch lr_min is: {lr_min}")
            learn.fit_one_cycle(1, lr_min, moms=(0.8,0.7,0.8), cbs=cbs)
        else:
            learn.fit_one_cycle(1, 2e-3, moms=(0.8,0.7,0.8), cbs=cbs)
        learn.save(f"{MODEL_NAME}_1_EPOCH")


    # Continue in all layers
    learn = learn.load(f"{MODEL_NAME}_1_EPOCH")
    
    if FIT_LM_FINE_TUNE:
        if LM_LR_FIND_FINE_TUNE:
            lr_min, lr_steep = learn.lr_find()

        tensorboard=URLs.LOCAL_PATH/'tmp'/'runs'/f"{MODEL_NAME}_FINE_TUNE"
        cbs=TensorBoardCallback(tensorboard, trace_model=False)
        
        learn.unfreeze()
        if lr_min:
            print(f"LM Fine Tune lr_min is: {lr_min}")
            learn.fit_one_cycle(10, lr_min, moms=(0.8,0.7,0.8), cbs=cbs)
        else:
            learn.fit_one_cycle(10, 2e-3, moms=(0.8,0.7,0.8), cbs=cbs)

        learn.save(f"{MODEL_NAME}_FINE_TUNE")
        learn.save_encoder(f"{MODEL_NAME}_FINE_TUNE_ENCODER")

    
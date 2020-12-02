#%%
from data.external import Config, path
from data.core import (SubWordTok, Numericalize, 
Datasets, LMDataLoaderX, tensor, language_model_learner)
import pandas as pd
from fastcore.foundation import L
from fastai.text.models.awdlstm import AWD_LSTM
from fastai.metrics import Perplexity, accuracy
from fastai.callback.fp16 import *
from fastai.callback.schedule import fine_tune, lr_find, fit_one_cycle



FNAME = "ct-gov"


awd_lstm_lm_config = dict(
        emb_sz=400, n_hid=1152, n_layers=3, pad_token=1, bidir=False, output_p=0.1,
        hidden_p=0.15, input_p=0.25, embed_p=0.02, weight_p=0.2, tie_weights=True, out_bias=True)


if __name__ == "__main__":

    # Intiate Configuration (dir's)
    Config()
    
    # Path to the data
    path_data = path(FNAME, c_key="data")
    path_data_lm = path_data/'clinical_trial_descriptions_sub.txt'

    # Load lm data
    df = pd.read_csv(path_data_lm, sep="\t", header=None)
    df.columns = ["text"]

    # Init the tokenizer, train on all 'items' and save the tokenizer in cache_dir
    tok = SubWordTok(cache_dir=path_data, items=df["text"].tolist())

    # Setup Numericalizer with 'dsets' and save the numericalizer in cache_dir
    num = Numericalize(
        cache_dir=path_data,
        min_freq=1,
        dsets=tok(df["text"].tolist()[:2000])
    )
    vocab = num.return_vocab()

    # Call tokenizer and numericalizer
    text = [
        "The investigators aimed to compare the block characteristics of the single operator 'jedi grip' technique and the conventional double operator technique.",
        "The purpose of the study is to characterize the safety and pharmacokinetic (PK) profile of UCB6114.",
        "The primary objective of the study aims to evaluate serological assays of virus Covid-19.",
        "The purpose of this study is to identify the therapeutic effects of family workshops on speech and language developmentally delayed children and their family",
        "A Phase 1 Study to Compare the Safety, Pharmacokinetics and Pharmacodynamics of HIP1802 to HGP1705 in Healthy Volunteers",
        "The purpose of this randomized cross-over clinical trial is to examine the effects of Mediterranean diet based intervention on inflammation, metabolic risk and microbiome in patients with dyslipidemia.",
        "fixation of FGG with sutures alone is not sufficient,we use cyanoacrylate beside sutures for fixation",
        "Evaluate the effectiveness of mesh reinforcement in high-risk patients to prevent incisional hernia.",
        "Dual objectives of increased efficacy compared to currently available SoC RA drugs and maintaining a favourable benefit - risk relationship.",
        "This study was conducted on neonates needing intubation; Group A,: the ETT insertion depth was estimated according to the OHL method. Group B,: the ETT insertion depth was estimated according to the 7-8-9 method."
    ]
    text = df["text"].tolist()[:5]

    # call the tokenizer
    # for t in tok(text):
    #     print(t)

    # call the numericalizer based on tokenizer output
    x=[num.encode(t) for t in tok(text)]
    # print(x)

    bs=8
    sl=4
    ints = L([0,1,2,3,4],[5,6,7,8,9,10],[11,12,13,14,15,16,17,18],[19,20],[21,22]).map(tensor)
    ints = x
    dl = LMDataLoaderX(ints, bs=bs, seq_len=sl, verbose=False)
    for x,y in dl:
        print(f'This is x: {x}')
        for tx in x:
            print(f'This is x decoded: {num.decode(tx)}')
        print(f'This is y: {y}')
        for ty in y:
            print(f'This is y decoded: {num.decode(ty)}')


    learn = language_model_learner(
        dls=dl,
        vocab=vocab,
        arch=AWD_LSTM,
        config=awd_lstm_lm_config,
        pretrained=False,
        drop_mult=0.3,
        metrics=[accuracy, Perplexity()]
        ).to_fp16()
    print(learn.model)

    
# %%

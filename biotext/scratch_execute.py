#%%
from data.external import Config, path
from data.core import (SubWordTok, Numericalize, 
Datasets, LMDataLoaderX, tensor)
import pandas as pd
from fastcore.foundation import L

FNAME = "ct-gov"

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

    # Call tokenizer and numericalizer
    text = [
        "operator technique is not promising and therefore clinically irrelevant",
        "this is not a very long effort in this assessment",
        "this is a hba1c above 3"
    ]
    text = df["text"].tolist()[:2]

    # call the tokenizer
    # for t in tok(text):
    #     print(t)

    # call the numericalizer based on tokenizer output
    x=[num.encode(t) for t in tok(text)]
    # print(x)

    bs=3
    sl=3
    ints = L([0,1,2,3,4],[5,6,7,8,9,10],[11,12,13,14,15,16,17,18],[19,20],[21,22]).map(tensor)
    # ints = x
    dl = LMDataLoaderX(ints, bs=bs, seq_len=sl)
    for x,y in dl:
        print(f'This is x: {x}')
        for tx in x:
            print(f'This is x decoded: {num.decode(tx)}')
        print(f'This is y: {y}')
        for ty in y:
            print(f'This is y decoded: {num.decode(ty)}')

    # Abstract both into a class and do behind the scene with __getitem__
    # ds = Datasets(items=text, tok=tok, num=num)
    # for i in range(len(ds)):
    #     print(ds[i])
    

    
# %%

#%%
from data.external import Config, path
from data.core import SubWordTok, Numericalize
import pandas as pd

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

    # Init the tokenizer, train and save
    tok = SubWordTok(cache_dir=path_data, items=df["text"].tolist())

    # Setup Numericalizer (with desired min f freq)
    num = Numericalize(
        cache_dir=path_data,
        min_freq=1,
        dsets=tok(df["text"].tolist()[:2000])
    )

    # Call tokenizer and numericalizer
    text = "operator technique is not promising and therefore clinically irrelevant".split()
    x=[num.encode(t) for t in tok(text)]
    print(x)


    
# %%

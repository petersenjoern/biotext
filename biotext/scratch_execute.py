#%%
from data.external import Config, path
from data.core import SubWordTok, Numericalize, Datasets
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
        "this is not",
        "this is a hba1c above 3"
    ]
    text = df["text"].tolist()[:3]

    # call the tokenizer
    for t in tok(text):
        print(t)

    # call the numericalizer based on tokenizer output
    x=[num.encode(t) for t in tok(text)]
    print(x)


    # Abstract both into a class and do behind the scene with __getitem__
    ds = Datasets(items=text, tok=tok, num=num)
    for i in range(len(ds)):
        print(ds[i])


    
# %%

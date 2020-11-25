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
    tok = SubWordTok(cache_dir=path_data)
    tok.setup(df["text"].tolist(), retrain=False)
    # for t in tok("No translation can expect to equal, much less to excel, the original. The excellence of a translation can only be judged by noting how far it".split()):
    #     print(t)

    num = Numericalize(min_freq=1)
    num.setup(tok(df["text"].tolist()[:2000]))

    text = "operator technique is not promising and therefore clinically irrelevant".split()

    x=[num.encode(t) for t in tok(text)]
    print(x)


    
# %%

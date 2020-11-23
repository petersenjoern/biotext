#%%
from data.external import Config, path
from data.core import SubWordTok
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
    tok.setup(df["text"].tolist())
    for t in tok("No translation can expect to equal, much less to excel, the original. The excellence of a translation can only be judged by noting how far it"):
        print(t)
# %%

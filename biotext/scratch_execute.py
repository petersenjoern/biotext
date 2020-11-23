#%%
from data.external import Config, path
from data.core import SubWordTok
from pathlib import Path
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



cache_dir = Path.cwd()
raw_text_path = cache_dir/'data'/'botchan.txt'

tok = SubWordTok(cache_dir)
tok.setup(raw_text_path)
for t in tok("No translation can expect to equal, much less to excel, the original. The excellence of a translation can only be judged by noting how far it"):
    print(t)
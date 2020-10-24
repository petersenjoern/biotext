"""
Various generic utilities
"""

import pickle
import pathlib
import pandas as pd

NL = '\n'

def resolve_path(path, filename):
    "resolve path with pathlib"

    try:
        path=pathlib.Path(path)
        path_full=path.joinpath(filename)
    except:
        path_full=path + filename

    return path_full

def pickle_save(obj=None, path=None):
    "Function to objects to pickle"

    with open(path, 'wb') as output:
        pickle.dump(obj, output)

def pickle_load(path=None):
    "Function for loading pickle files"

    with open(path, 'rb') as f:
        return pickle.load(f)

def TextDataLoadersInspector(dls, vocab_len=30, n_vocab_items=10):
    "Inspect TextDataLoaders"

    # Vocab information
    voc = dls.vocab[:vocab_len]
    dls_counter_s = pd.Series({key: value for key, value in dls.counter.items()})
    vocab_items_nlargest=dls_counter_s.nlargest(n_vocab_items)
    vocab_items_nsmallest=dls_counter_s.nsmallest(n_vocab_items)

    # Batch information
    # 0 = training, 1 = validation
    bt_train = dls[0].one_batch() # contains (x[batches],y[batches])
    x,y = bt_train # contains ([batches],[batches])
    bt_train_decoded = dls.decode_batch(bt_train) # contains batches[(x,y)]

    print("*" * 80)
    print(f"first {vocab_len} vocab elements:{NL}{voc}")
    print("*" * 80)
    print(f"top {n_vocab_items} vocab items:{NL}{vocab_items_nlargest}")
    print("*" * 80)
    print(f"bottom {n_vocab_items} vocab items:{NL}{vocab_items_nsmallest}")
    print("*" * 80)


    print("One Batch has structure: tuple(x[bs,seq_len],y[bs,seq_len])")
    print(f"x has shape of:{NL}{x.shape}")
    print(f"y has shape of:{NL}{y.shape}")
    print(f"{x.shape[0]} == bs, {x.shape[1]} == seq_len")
    print("-" * 80)
    print(f"x[0] contains num for first batch of x:{NL}{x[0]}")
    print(f"x[0] nums decoded:{NL}{bt_train_decoded[0][0]}")
    print("+" * 80)
    print(f"y[0] contains num for first batch of y:{NL}{y[0]}")
    print(f"x[0] nums decoded:{NL}{bt_train_decoded[0][1]}")
    print("-" * 80)

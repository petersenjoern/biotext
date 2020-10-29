#%%

import torch
from fastai.text.all import *
from utils import TextDataLoadersInspector


#%%
## Get the data for x
path = untar_data(URLs.IMDB)
files = get_text_files(path, folders=["train", "test"])
files[0]

#%%
## Get data for y and convert y to idx (numericalize)
lbls = files.map(Self.parent.name()).unique()
v2i = lbls.val2idx()
print(v2i)

#%%

# Build a Dataset for PyTorch (has to support indexing, e.g. __getitem__ and __len__)
class Dataset:
    """Dataset for Pytorch (has to support indexing of data, hence __getitem__ and __len__).
    Input data is a list of file paths"""
    
    def __init__(self, fns):
        self.fns = fns
        
    def __len__(self): 
        return len(self.fns)
    
    def __getitem__(self, i):
        txt = self.fns[i].open().read()
        return txt

    
def random_splitting(valid_pct=0.2, seed=None):
    "Create function that splits items between train/val with valid_pct randomly."
    def _inner(o):
        if seed is not None: torch.manual_seed(seed)
        rand_idx = L(int(i) for i in torch.randperm(len(o)))
        cut = int(valid_pct * len(o))
        return rand_idx[cut:],rand_idx[:cut]
    return _inner


#%%
train, valid = random_splitting(valid_pct=0.1)(files) 
train[0:5]


#%%
valid_ds = Dataset(valid)
valid_ds[0]

    # path = untar_data(URLs.IMDB)
    # print(path.ls())

    # files = get_text_files(path)
    # txts = L(o.open().read() for o in files[:2000])
    # print(txts[0])

    # tok = Tokenizer.from_folder(path)
    # tok.setup(txts)
    # toks = txts.map(tok)
    # print(tok(txts[0]))

    # num = Numericalize
    # tfms = Pipeline([tok, num])
    # t = tfms(txts[0])
    # print(t)
    

    # tls = TfmdLists(files, [Tokenizer.from_folder(path), Numericalize])
    # print(tls[0])

    # dls = TextDataLoaders.from_folder(path, valid='test', seq_len=80)
    # print(dls.show_batch())
    # learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)
    # TextDataLoadersInspector(dls)
#%%

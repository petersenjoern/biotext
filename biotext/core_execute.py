#%%
from fastai.text.all import *
from core import (PipelineX, CategorizeX, TfmdListsX, DatasetsX, TokenizerX)
from utils import TextDataLoadersInspector

#%%

path = untar_data(URLs.IMDB)
files = get_text_files(path, folders=["train", "test"])

txts_inputs = L(o.open().read() for o in files[:2000])
lbls_targets = files.map(parent_label) 
#%%

# Setup of the Tokenizer on txts_inputs
tok = Tokenizer.from_folder(path)
tok.setup(txts_inputs)
toks = txts_inputs.map(tok)

# Setup Numericalize based on Tokenizer
num = Numericalize()
num.setup(toks)

#%%
# Setup Categorize based on lbls_targets
cat = CategorizeX()
cat.setup(lbls_targets) # non-unique list of targets
cat.vocab, cat('neg'), cat.decodes(cat('neg')) # will be unique and TensorCategory
#%%

# Pipeline requires that the pipeline elements have been set up
# so the encodes/decodes functions of the object can be called
# When calling Pipeline(), each Transform (e.g. Tokenizer (inherits from
# from Transform)) will execute encodes()
# When calling Pipeline.decode, each Transform (in reversed order)
# will execute decodes() on the Transform object 
tfms=PipelineX([tok, num])

t = tfms(txts_inputs[0]) #executes .encodes() for each transformer
d = tfms.decode(t)[:100] #executes .decodes() for each transformer, reversed order

# => To abstract the Setup of the Transformers further away,
# use TfmdLists (will automatically call the setup method
# for each Transform in order)

# %%
# TfmdLists requires from Pipeline the setup() and __call__(o) function
# First thing TfmdLists does is to call its own def setup()
# TfmdLists setup() will go to Pipeline and call its setup() where its beeing looped
# over compose_tfmsx() which executes encodes() for each of the Pipeline elements.
# The transforms are in order, thus the items will be transformed based on
# all the previous Transforms.
# In summary, it does the above manual .setup() and .encodes() steps

# However, in addition, it will add the transformed data .train / .valid (if valid is provided)

tls_x = TfmdListsX(
    files,
    [Tokenizer.from_folder(path), Numericalize]
)
tls_x.train[0].shape
# %%
# You need to pas the indices of the elements that are in the training set
# and in the indices of the elements that are in the validation set.

cut=int(len(files)*0.8)
splits = [list(range(cut)), list(range(cut, len(files)))]

tls_x = TfmdListsX(
    files,
    [Tokenizer.from_folder(path), Numericalize],
    splits=splits
)
print(f"train_x: {tls_x.train[0][:20]}, valid_x: {tls_x.valid[0][:20]}")
# %%
tls_y = TfmdListsX(
    files,
    [parent_label, Categorize()],
    splits=splits
)
print(f"train_y: {tls_y.train[:20]}, valid_y: {tls_y.valid[:20]}")

# %%
## Use Datasets (get both x and y in parallel transformed)
cut=int(len(files)*0.8)
splits = [list(range(cut)), list(range(cut, len(files)))]

tfms_x = [Tokenizer.from_folder(path), Numericalize]
tfms_y = [parent_label, Categorize()]
dsets = DatasetsX(files, [tfms_x, tfms_y], splits=splits)
x,y = dsets.train[0]
print(x[:20], y)


# %%
## Explore different settings of dataloaders (after_item, before_batch, after_batch)
dls = dsets.dataloaders(dl_type=SortedDL, before_batch=pad_input)
bt_train = dls.loaders[0].one_batch()
x,y = bt_train

#%%
## There are limitations with DatasetsX (not not propagating all values down)
## Instead, this will work, however, the seq_len is not recognised

tfms = [[Tokenizer.from_folder(path), Numericalize], [parent_label, Categorize]]
files = get_text_files(path, folders=['train', 'test'])
splits = GrandparentSplitter(valid_name='test')(files)
dsets = Datasets(files, tfms, splits=splits)
dls = dsets.dataloaders(bs=8, seq_len=80, dl_type=SortedDL, before_batch=pad_input)
TextDataLoadersInspector(dls)
## Have probably to use the DataBlock API with TextBlock to get the seq_len in

#%%
## Use TextBlock API to validate above hypothesis
dls = DataBlock(
    blocks=(TextBlock.from_folder(path), CategoryBlock),
    get_y=parent_label,
    get_items=partial(get_text_files, folders=["train", "test"]),
    splitter=GrandparentSplitter(valid_name="test")
).dataloaders(path, bs=8, seq_len=80)
TextDataLoadersInspector(dls)
## This is rejecting the hypothesis, seq_len is also not applied

#%%
# Go back to the Tokenizer and explore its setup and encode
# Create TokenizerX
# Setup of the Tokenizer on txts_inputs
# Have to adjust the tokenizer to parse scientific biomedical text
# rather than just the en internet text :)
tok = TokenizerX.from_folder(path)
tok.setup(txts_inputs)
toks = txts_inputs.map(tok)

#%%
# next steps, align tokens and labels
text = "Since PLETAL is extensively metabolized by cytochrome P-450 isoenzymes, caution should be exercised when PLETAL is coadministered with inhibitors of C.P.A. such as ketoconazole and erythromycin or inhibitors of CYP2C19 such as omeprazole."
annotations = [
    {"start": 6, "end": 13, "tag": "drug"},
    {"start": 43, "end": 60, "tag": "drug"},
    {"start": 105, "end": 112, "tag": "drug"},
    {"start": 164, "end": 177, "tag": "drug"},
    {"start": 181, "end": 194, "tag": "drug"},
    {"start": 211, "end": 219, "tag": "drug"},
    {"start": 227, "end": 238, "tag": "drug"}
]

for anno in annotations:
    # Show our annotations
    print (text[anno['start']:anno['end']],anno['tag'])

#%%

from transformers import BertTokenizerFast, BatchEncoding
from tokenizers import Encoding
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased') # Load a pre-trained tokenizer
tokenized_batch : BatchEncoding = tokenizer(text)
tokenized_text :Encoding  =tokenized_batch[0]

#%%




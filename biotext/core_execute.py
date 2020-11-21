#%%
import ast
from fastai.text.all import *
from core import (PipelineX, CategorizeX, TfmdListsX, DatasetsX, TokenizerX, SubWordTok)
from utils import TextDataLoadersInspector
from data.external import Config, path

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

# Intiate Configuration (dir's)
Config()

# Path to the data
path_data = path("ct-gov", c_key="data")
path_data_ner = path_data/'train_processed_medical_ner_biotext.csv'

df = pd.read_csv(path_data_ner)
df.head()
df.iloc[8]

#%%
txts_inputs = df["x"][:200]
txts_inputs

#%%
from transformers import BertTokenizerFast, BatchEncoding
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

class TransformersTokenizer(Transform):
    def __init__(self, tokenizer): self.tokenizer = tokenizer
    def encodes(self, x): 
        toks = self.tokenizer.tokenize(x)
        return tensor(self.tokenizer.convert_tokens_to_ids(toks))
    def decodes(self, x): return TitledStr(self.tokenizer.decode(x.cpu().numpy()))


cut=int(len(txts_inputs)*0.8)
splits = [list(range(cut)), list(range(cut, len(txts_inputs)))]

tls_x = TfmdListsX(
    txts_inputs,
    TransformersTokenizer(tokenizer),
    splits=splits, dl_type=LMDataLoader
)
print(f"train_x: {tls_x.train[0]}, valid_x: {tls_x.valid[0]}")
bs,sl = 8,1024
dls = tls_x.dataloaders(bs=bs, seq_len=sl)

#%%
# next steps, align tokens and labels
# similar to: https://github.com/LightTag/sequence-labeling-with-transformers/blob/master/notebooks/how-to-align-notebook.ipynb
text = "end-stage organ disease or medical condition with subsequent vision loss ( e.g. , diabetes , stroke )"

for anno in y:
    # Show our annotations
    print(text[anno['start']:anno['end']],anno['label'])


#%%
aligned_labels = ["O"]*len(x) # Make a list to store our labels the same length as our tokens
for anno in (y):
    for char_ix in range(anno['start'],anno['end']):
        


#%%

from transformers import BertTokenizerFast, BatchEncoding
from tokenizers import Encoding
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased') # Load a pre-trained tokenizer
tokenized_batch : BatchEncoding = tokenizer(text)
tokenized_text :Encoding  =tokenized_batch[0]

#%%

tokens = tokenized_text.tokens
aligned_labels = ["O"]*len(tokens) # Make a list to store our labels the same length as our tokens
for anno in (annotations):
    for char_ix in range(anno['start'],anno['end']):
        token_ix = tokenized_text.char_to_token(char_ix)
        if token_ix is not None: # White spaces have no token and will return None
            aligned_labels[token_ix] = anno['label']
for token,label in zip(tokens,aligned_labels):
    print (token,"-",label)


# %%
# Accounting For Multi Token Annotations
# BIOLU scheme, which will indicate if a token is the begining, inside, last token in an annotation or if it is not part of an annotation or if it is perfectly aligned with an annotation.
def align_tokens_and_annotations_bilou(tokenized: Encoding, annotations):
    tokens = tokenized.tokens
    aligned_labels = ["O"] * len(
        tokens
    )  # Make a list to store our labels the same length as our tokens
    for anno in annotations:
        annotation_token_ix_set = (
            set()
        )  # A set that stores the token indices of the annotation
        for char_ix in range(anno["start"], anno["end"]):

            token_ix = tokenized.char_to_token(char_ix)
            if token_ix is not None:
                annotation_token_ix_set.add(token_ix)
        if len(annotation_token_ix_set) == 1:
            # If there is only one token
            token_ix = annotation_token_ix_set.pop()
            prefix = (
                "U"  # This annotation spans one token so is prefixed with U for unique
            )
            aligned_labels[token_ix] = f"{prefix}-{anno['label']}"

        else:

            last_token_in_anno_ix = len(annotation_token_ix_set) - 1
            for num, token_ix in enumerate(sorted(annotation_token_ix_set)):
                if num == 0:
                    prefix = "B"
                elif num == last_token_in_anno_ix:
                    prefix = "L"  # Its the last token
                else:
                    prefix = "I"  # We're inside of a multi token annotation
                aligned_labels[token_ix] = f"{prefix}-{anno['label']}"
    return aligned_labels


labels = align_tokens_and_annotations_bilou(tokenized_text, annotations)
for token, label in zip(tokens, labels):
    print(token, "-", label)
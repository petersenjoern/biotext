#%%
from fastai.text.all import *
from core import PipelineX, CategorizeX

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


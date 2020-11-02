#%%
from fastai.text.all import *
from core import PipelineX, CategorizeX, TfmdListsX

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
# over compose_tfmsx() which executes encodes() for each of the Pipeline elements
# In summary, it does the above manual .setup() and .encodes() steps
# However, in addition, it will add the transformed data .train / .valid (if valid is provided)


tls = TfmdListsX(files, [Tokenizer.from_folder(path), Numericalize])
tls.train[0].shape
# %%

# %%

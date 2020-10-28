
from fastai.text.all import *
from utils import TextDataLoadersInspector


if __name__ == "__main__":
    
    path = untar_data(URLs.IMDB)
    print(path.ls())

    files = get_text_files(path, folders=["train"])
    txts = L(o.open().read() for o in files[:2000])
    print(txts[0])

    tok = Tokenizer.from_folder(path)
    # tok.setup(txts)
    # toks = txts.map(tok)
    # print(tok(txts[0]))

    num = Numericalize


    tfms = Pipeline([tok, num])
    t = tfms(txts[0])
    print(t)
    

    # tls = TfmdLists(files, [Tokenizer.from_folder(path), Numericalize])
    # print(tls[0])

    # dls = TextDataLoaders.from_folder(path, valid='test', seq_len=80)
    # print(dls.show_batch())
    # learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)
    # TextDataLoadersInspector(dls)
"""
Losely decoupled file for exploring library functionalities
"""
from data.external import Config, path
from core import tokenize_df, SentencePieceTokenizer, Tokenizer
from fastcore.foundation import L, first
import pandas as pd

FNAME: str = "wikitext-2"
VOCAB_SZ: int = 2000
TEXT_COL: str = "text"

def apply_sentence_piecer(df, text_col:str='text', vocab_sz:int=200, print_len:int=40):
    "Apply SentencePieceTokenizer"

    txt = L(df[text_col].tolist())
    sp = SentencePieceTokenizer(vocab_sz)
    sp.setup(txt)
    return next(iter(sp(txt[:print_len])))


def func_tokenize_df_results(df, tok, text_col:str='text'):
    "Function to apply Tokenizer to df and get tokenizer results summary"

    out,cnt=tokenize_df(df=df, text_cols=text_col, tok=tok)
    return out,cnt


def wrapper_tokenizer_class(df, tok, text_col:str='text', print_len:int=40):
    "Wrapper around Tokenizer (easy to use e.g. from_df)"

    toke=Tokenizer.from_df(df=df, text_cols=text_col, tok=tok)
    toke.tok.setup(L(toke.kwargs['df'][text_col].tolist()))
    tokenizer_res = first(toke.tok(L(toke.kwargs['df'][text_col].tolist()[:print_len])))
    toke_obj = toke.__dict__
    return tokenizer_res, toke_obj



if __name__ == "__main__":

    # Intiate Configuration (dir's)
    Config()

    # Path to the data
    path_wikitext = path(FNAME, c_key="data")
    path_wikitext_train = path_wikitext/'train.csv'
    path_wikitext_valid = path_wikitext/'test.csv'

    # Load data
    df_train = pd.read_csv(path_wikitext_train, header=None)
    df_valid = pd.read_csv(path_wikitext_valid, header=None)
    df_all = pd.concat([df_train, df_valid])
    df_all.columns = [TEXT_COL]

    vocab = apply_sentence_piecer(df_all, vocab_sz=VOCAB_SZ)
    print(vocab[:20])

    out,cnt = func_tokenize_df_results(df_all, tok=SentencePieceTokenizer(vocab_sz=VOCAB_SZ))
    print(out)
    print(cnt)

    res, toke_obj = wrapper_tokenizer_class(df_all, tok=SentencePieceTokenizer(vocab_sz=VOCAB_SZ), print_len=20)
    print(res)
    print(toke_obj)




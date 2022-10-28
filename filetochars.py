from fileinput import filename
import io

import numpy as np



def file_to_text(filename):
    with io.open(filename, encoding="utf-8") as f:
        text = f.read().lower()
    text = text.replace("\n", " ")  # We remove newlines chars for nicer display
    print("Corpus length:", len(text))
    return text

    

def text_to_chars(text):
    chars = sorted(list(set(text)))
    print("Total chars:", len(chars))
    return chars
def char_indices(chars):
    char_indices = dict((c, i) for i, c in enumerate(chars))
    return char_indices
def indices_char(chars):
    indices_char = dict((i, c) for i, c in enumerate(chars))
    return indices_char




   

        
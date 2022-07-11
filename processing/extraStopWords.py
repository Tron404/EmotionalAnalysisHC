import pickle
import numpy as np
import pandas as pd
import re

file = open("global_dictionary.pickle", "rb")
word_dictionary = pickle.load(file)

text = {key:value for key, value in word_dictionary if value > 1}
text = set(text)
text = " ".join(text)
text = re.sub(r"[^\w]+", " ", text)
text = re.split(r" ", text)
text = set(text)

file = open("singletonFreeSet.pickle", "wb")

pickle.dump(text, file)

    
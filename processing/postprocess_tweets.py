import pandas as pd
import numpy as np
import pickle
import os

singletonFreeSet = pickle.load(open("singletonFreeSet.pickle", "rb"))

data = pd.read_csv("all_features.csv", index_col=False, skip_blank_lines=True, dtype=str, encoding="")
data["text"] = data["text"].str.replace(r"[\[\]]", "", regex=True).replace("'", "", regex=True).replace(r" ", "", regex=True).str.split(",")

for text in data["text"]:
    aux = text
    for word in aux:
        if word not in singletonFreeSet:
            text.remove(word)

print(len(data))

# data["text"] = [" ".join(text) for text in data["text"]]
data.to_csv("all_features_clean.csv", index=False)

import numpy as np
import pandas as pd
import os

feature_csv = os.listdir("features")
file_total = open("all_features.csv", "wb")
all_features = pd.read_csv("features/" + feature_csv[0], sep=",", index_col=False, skip_blank_lines=True, encoding="", dtype=str)

for file in feature_csv[1:]:
    print(file)

    aux = pd.read_csv("features/" + file, sep=",", index_col=False, skip_blank_lines=True, encoding="", dtype=str)
    if(len(list(aux.columns))<3):
        aux.insert(0, "id", -np.ones((len(aux),1), dtype=np.int32))

    all_features = pd.concat([all_features, aux])

all_features.to_csv(file_total, index=False)
print(len(all_features))
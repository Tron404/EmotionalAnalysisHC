import pandas as pd
import numpy as np
from sklearn.utils import shuffle 

file = "all_features_clean.csv"

random_state = 80
epoch_number = 25

def readData(file):
    data = pd.read_csv(file, index_col=False, skip_blank_lines=True, dtype=str, encoding="", usecols=["text", "TruthValue"])
    data["text"] = data["text"].str.replace(r"[\[\]]", "", regex=True).replace("'", "", regex=True).replace(r" ", "", regex=True).str.split(",")
    data["text"] = [" ".join(text) for text in data["text"]]
    
    return data

data = readData(file)
new_data = data[data["TruthValue"] == "0"]
sampled_data = data[data["TruthValue"] == "1"].sample(len(new_data), random_state=random_state)

new_data = pd.concat([new_data, sampled_data])
new_data = shuffle(new_data)
new_data = new_data.reset_index(drop=True)

print(new_data)
print(len(new_data))

import pandas as pd
import numpy as np

file = "data_set_tweet_user_features.csv"
columns = ["tweet__id", "tweet__text", "tweet__fake"]

data = pd.read_csv(file, sep=";", index_col=False, skipinitialspace=True, skip_blank_lines=True, dtype=str, usecols=columns)

new_csv = pd.DataFrame()
new_csv["id"] = data[columns[0]]
new_csv["text"] = data[columns[1]]
new_csv["TruthValue"] = data[columns[2]]

print(new_csv.head())

new_csv.to_csv("labelled_data_weaksupervision.csv", index=False)
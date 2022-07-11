import numpy as np
import pandas as pd

data_csv = pd.read_csv("labelled_data_weaksupervision.csv", sep=",", index_col=False, skipinitialspace=True, skip_blank_lines=True, dtype=str)

for idx in range(len(data_csv)):
    if data_csv.at[idx, "TruthValue"] == "0":
        data_csv.at[idx, "TruthValue"] = "1"
    else:
        data_csv.at[idx, "TruthValue"] = "0"

data_csv.to_csv("labelled_data_weaksupervision_processed_invertedlabels.csv", index=False)
import pandas as pd
import os
# import plotly.graph_objects as go
import numpy as np

total_tweets = 0
true_tweets = 0
false_tweets = 0
values = []
for file in os.listdir("features"):
    if file.endswith(".csv") and file != "aux.csv":
        cvs_data = pd.read_csv("features/" + file, sep=",", index_col=False, skipinitialspace=True, skip_blank_lines=True, dtype=str, usecols=["TruthValue"])
        
        total_tweets += len(cvs_data)
        aux_t = len(cvs_data[cvs_data["TruthValue"] == "1"])
        aux_f = len(cvs_data[cvs_data["TruthValue"] == "0"])
        true_tweets += aux_t
        false_tweets += aux_f

        values.append([aux_f, aux_t, aux_t+aux_f])

        print(f"File {file} has {len(cvs_data)} total tweets ==== {aux_t} true tweets ==== {aux_f} false tweets")
        print("-------------------")

print(f"There are /{total_tweets}/ tweets in total, of which /{true_tweets}/ are true and /{false_tweets}/ are false")
print(f"There is a ratio of {true_tweets/total_tweets}% true tweets and {false_tweets/total_tweets}% false tweets")

names = ["Shahi et al. - 2021", "Hayawi et al. - 2022", "Shahi and Nandini - 2020", "Alam et al. - 2021", "Cheng et al. - 2021 | news", "Helmstetter and Paulheim - 2021", "Cui and Lee - 2020 | tweets/news", "Elhadad et al. - 2020", "Cui and Lee - 2020 | tweets/claims", "Cheng et al. - 2021 | tweets", " "]

values_f = [v[0] for v in values]
values_f.append("<b>" + str(np.sum(values_f)) + "</b>")
values_t = [v[1] for v in values]
values_t.append("<b>" + str(np.sum(values_t)) + "</b>")
values_total = [v[2] for v in values]
values_total.append("<b>" + str(np.sum(values_total)) + "</b>")

# fig = go.Figure(data=[go.Table(header=dict(values=['Source of dataset', 'Number of false entries', "Number of true entries", "Number of total entries"], align="center"),
#                  cells=dict(values=[names, values_f, values_t, values_total], align="center"), domain=dict(x=[0,0.5]))])


print(pd.DataFrame(dict(values=[names, values_f, values_t, values_total])))

# fig.show()
# fig.write_image("table_raw.png", width=1800, height=500, scale=2)
import collections
import math
import pandas as pd
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from nrclex import NRCLex

file = "all_features_clean.csv"
size = 0.5

def zero_list():
    return [0,0]

def readData(file):
    data = pd.read_csv(file, index_col=False, skip_blank_lines=True, dtype=str, encoding="", usecols=["text", "TruthValue"])
    data["text"] = data["text"].str.replace(r"[\[\]]", "", regex=True).replace("'", "", regex=True).replace(r" ", "", regex=True).str.split(",")
    data["text"] = [" ".join(text) for text in data["text"]]

    return data

data = readData(file)
data = data.sample(math.floor(len(data)*size))
 
emotion = []
for text in data["text"]:
    text_obj = NRCLex(text)
    emotion.append(text_obj.top_emotions[0][0])

data["emotion"] = emotion
binary = [0,0]
emotion_dict = collections.defaultdict(zero_list)

for truth, emo in zip(data["TruthValue"], data["emotion"]):
    emotion_dict[emo][int(truth)] += 1

pickle.dump(emotion_dict, open(f"test_{size}.pickle", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
df = pd.DataFrame(emotion_dict)
truth_values = df.values

fig = go.Figure()

print(df)

fig.add_trace(go.Scatterpolar(r=np.log(truth_values[0]), theta=list(df.columns), fill="toself", name="False"))
fig.add_trace(go.Scatterpolar(r=np.log(truth_values[1]), theta=list(df.columns), fill="toself", name="True"))

fig.update_layout(legend = dict(font = dict(family = "Courier", size = 15, color = "black")),
                  legend_title = dict(font = dict(family = "Courier", size = 10, color = "blue")),
                  autosize = False,
                  width=500,
                  height=450, title = "Distribution on a log scale of the emotions<br>across true and false news</br>", title_x=0.5)

fig.write_image(f"emotion_dist_{size}.png", width=500, height=450, scale=2)

fig.show()


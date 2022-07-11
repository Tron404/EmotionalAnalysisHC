import pandas as pd
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import pickle
import matplotlib.pyplot as plt

def zero_list():
    return [0,0]

df = pickle.load(open("test.pickle", "rb"))

df = pd.DataFrame(df)
truth_values = df.values

fig = go.Figure()

print(df)

fig.add_trace(go.Scatterpolar(r=np.log(truth_values[0]), theta=list(df.columns), fill="toself", name="False"))
fig.add_trace(go.Scatterpolar(r=np.log(truth_values[1]), theta=list(df.columns), fill="toself", name="True"))

fig.update_layout(legend = dict(font = dict(family = "Courier", size = 30, color = "black")),
                  legend_title = dict(font = dict(family = "Courier", size = 20, color = "blue")),
                  autosize = False,
                  width=500,
                  height=450, title = "Distribution on a log scale of the emotions<br>across true and false news</br>", title_x=0.5)

# fig.write_image("emotion_dist.png", width=500, height=450, scale=2)

fig.show()
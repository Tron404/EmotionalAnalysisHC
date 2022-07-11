import collections
import numpy as np
import pandas as pd
from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt

data = pd.read_csv("all_features_clean.csv", index_col=False, skip_blank_lines=True, dtype=str, encoding="", usecols=["text", "TruthValue"])
data["text"] = data["text"].str.replace(r"[\[\]]", "", regex=True).replace("'", "", regex=True).replace(r" ", "", regex=True).str.split(",")

truth_value = "1"
dict = collections.defaultdict(int)
for _, row in data.iterrows():
    if truth_value == str(row["TruthValue"]):
        for word in row["text"]:
            dict[word] += 1

# https://matplotlib.org/3.5.0/tutorials/colors/colormaps.html - colour maps
cloud = WordCloud(mode="RGBA", width=5500, height=5500, max_words=150,random_state=2, background_color="rgba(255, 255, 255, 0)", colormap="cool").generate_from_frequencies(dict)

plt.tight_layout(pad=0)
plt.figure(figsize=(30,30))
plt.axis("off")
plt.imshow(cloud, interpolation="spline16")
cloud.to_file(f"cld_{truth_value}.png")
# plt.show()

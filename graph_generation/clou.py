import pickle
import numpy as np
import pandas as pd
from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt

file = open("global_dictionary.pickle", "rb")
image_file = Image.open("hashtag-solid.jpg")
# mask = Image.new("RGB", image_file.size, (255, 255, 255))
# mask.paste(image_file, image_file)
mask = np.array(image_file)
word_dictionary = pickle.load(file)

text = {key:value for key, value in word_dictionary}

# https://matplotlib.org/3.5.0/tutorials/colors/colormaps.html - colour maps
cloud = WordCloud(mode="RGBA", width=5500, height=5500, max_words=150,random_state=2, background_color="rgba(255, 255, 255, 0)", colormap="cool").generate_from_frequencies(text)
# cool, winter, brg

plt.tight_layout(pad=0)
plt.figure(figsize=(30,30))
plt.axis("off")
plt.imshow(cloud, interpolation="spline16")
cloud.to_file("cld.png")
# plt.savefig('wordcloud.png', bbox_inches='tight', trasparent=True)
# plt.show()

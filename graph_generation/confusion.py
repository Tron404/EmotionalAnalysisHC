import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

def processConfMatrix(text, textemotion):
    text_conf = [t[0] for t in text]
    textemo_conf = [t[0] for t in textemotion]

    text_overall= [[0,0],[0,0]]
    textemo_overall = [[0,0],[0,0]]
    for tc, tec in zip(text_conf, textemo_conf):
        for row in range(len(tc)):
            for col in range(len(tc)):
                text_overall[row][col] += tc[row][col]
                textemo_overall[row][col] += tec[row][col]

    text_overall = np.divide(text_overall, len(SVM_text))
    textemo_overall = np.divide(textemo_overall, len(SVM_text))

    text_overall = text_overall.astype(np.int32)
    textemo_overall = textemo_overall.astype(np.int32)

    return text_overall, textemo_overall

SVM_text = pickle.load(open("SVM_text.pickle", "rb"))
SVM_text_emotion = pickle.load(open("SVM_textemotion.pickle", "rb"))
MNB_text = pickle.load(open("MNB_text.pickle", "rb"))
MNB_text_emotion = pickle.load(open("MNB_textemotion.pickle", "rb"))

SVM_text_overall, SVM_text_emotion_overall = processConfMatrix(SVM_text, SVM_text_emotion)
MNB_text_overall, MNB_text_emotion_overall = processConfMatrix(MNB_text, MNB_text_emotion)

fig, axes = plt.subplots(2,2, figsize=(15,15))

fig.suptitle("Confusion lattice for SVM and MNB across different features", fontsize=35)
cbar_ax = fig.add_axes([.93, 0.14, .03, 0.7])

vmax = np.max([SVM_text_overall.max(0), SVM_text_emotion_overall.max(0)])
vmin = np.min([SVM_text_overall.min(0), SVM_text_emotion_overall.min(0)])

########### SVM
hm_text = sns.heatmap(data = SVM_text_overall, ax=axes[0,0], annot=True, fmt="d", linewidths=0.5, cmap="flare", cbar_ax=None, cbar=None, vmax=vmax, vmin=vmin)
hm_text.set_xlabel("Predicted truth value", fontsize=13)
hm_text.xaxis.set_label_position("top")
hm_text.xaxis.tick_top()
hm_text.set_ylabel("Real truth value", fontsize=13)

hm_text = sns.heatmap(data = SVM_text_emotion_overall, ax=axes[1,0], annot=True, fmt="d", linewidths=0.5, cmap="flare", cbar_ax=cbar_ax, vmax=vmax, vmin=vmin)
hm_text.xaxis.tick_top()
hm_text.set_ylabel("Real truth value", fontsize=13)

########### MNB
hm_text = sns.heatmap(data = MNB_text_overall, ax=axes[0,1], annot=True, fmt="d", linewidths=0.5, cmap="flare", cbar_ax=None, cbar=None, vmax=vmax, vmin=vmin)
hm_text.set_xlabel("Predicted truth value", fontsize=13)
hm_text.xaxis.set_label_position("top")
hm_text.xaxis.tick_top()

hm_text = sns.heatmap(data = MNB_text_emotion_overall, ax=axes[1,1], annot=True, fmt="d", linewidths=0.5, cmap="flare", cbar_ax=cbar_ax, vmax=vmax, vmin=vmin)
hm_text.xaxis.tick_top()

fig.text(0.51, 0.02, 'Classifier', ha='center', fontsize=25)
fig.text(0.30, 0.06, 'SVM', ha='center', fontsize=20)
fig.text(0.72, 0.06, 'MNB', ha='center', fontsize=20)


fig.text(0.02, 0.5, 'Used Features', va='center', rotation='vertical', fontsize=25)
fig.text(0.06, 0.29, 'Text and emotion', va='center', rotation='vertical', fontsize=20)
fig.text(0.06, 0.71, 'Text', va='center', rotation='vertical', fontsize=20)

fig.savefig("confusion_lattice.png", dpi=200)


plt.show()
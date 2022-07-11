import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

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

def getF1_Precision_Recall(scores):
    true_neg = scores[0][0]
    false_neg = scores[1][0]
    false_pos = scores[0][1]
    true_pos = scores[1][1]


    precision = true_pos/(true_pos+false_pos)
    recall = true_pos/(true_pos+false_neg)

    F1 = (2*precision*recall)/(precision+recall)

    return round(F1, 3), round(precision, 3), round(recall, 3)

SVM_text = pickle.load(open("SVM_text.pickle", "rb"))
SVM_text_emotion = pickle.load(open("SVM_textemotion.pickle", "rb"))
MNB_text = pickle.load(open("MNB_text.pickle", "rb"))
MNB_text_emotion = pickle.load(open("MNB_textemotion.pickle", "rb"))

SVM_text_overall, SVM_text_emotion_overall = processConfMatrix(SVM_text, SVM_text_emotion)
MNB_text_overall, MNB_text_emotion_overall = processConfMatrix(MNB_text, MNB_text_emotion)

SVM_text_Stats, SVM_textemo_Stats = getF1_Precision_Recall(SVM_text_overall), getF1_Precision_Recall(SVM_text_emotion_overall)
MNB_text_Stats, MNB_textemo_Stats = getF1_Precision_Recall(MNB_text_overall), getF1_Precision_Recall(MNB_text_emotion_overall)

print(f"SVM === text:")
print(f"Precision: {SVM_text_Stats[1]} === Recall: {SVM_text_Stats[2]} === F1-score: {SVM_text_Stats[0]}")
print(f"SVM === text+emotions:")
print(f"Precision: {SVM_textemo_Stats[1]} === Recall: {SVM_textemo_Stats[2]} === F1-score: {SVM_textemo_Stats[0]}")

print("==========================")
print(f"MNB === text:")
print(f"Precision: {MNB_text_Stats[1]} === Recall: {MNB_text_Stats[2]} === F1-score: {MNB_text_Stats[0]}")
print(f"MNB === text+emotions:")
print(f"Precision: {MNB_textemo_Stats[1]} === Recall: {MNB_textemo_Stats[2]} === F1-score: {MNB_textemo_Stats[0]}")
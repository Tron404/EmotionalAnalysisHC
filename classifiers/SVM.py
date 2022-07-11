import time
import numpy as np
import pandas as pd
import pickle
import math

from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from nrclex import NRCLex

file = "all_features_clean.csv"

tfidf = TfidfVectorizer(stop_words="english")
encoder = LabelEncoder()

random_state = 80
epoch_number = 20

def readData(file):
    data = pd.read_csv(file, index_col=False, skip_blank_lines=True, dtype=str, encoding="", usecols=["text", "TruthValue"])
    data["text"] = data["text"].str.replace(r"[\[\]]", "", regex=True).replace("'", "", regex=True).replace(r" ", "", regex=True).str.split(",")
    data["text"] = [" ".join(text) for text in data["text"]]
    
    return data


def train_SVM(features, labels, type):
    print(f"Training an SVM of type: {type}")
    start_time = time.time()

    param_grid = {'C': [0.5, 1],
            'gamma': [0.01, 0.0001],
            'kernel': ['linear']}

    print("Dataset has been split!")
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=7)

    x_train = tfidf.fit_transform(x_train)
    x_test = tfidf.transform(x_test)

    y_train = encoder.fit_transform(y_train)
    y_test = encoder.fit_transform(y_test)

    grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3, cv=3)

    grid.fit(x_train, y_train)

    pred = grid.predict(x_test)

    score = accuracy_score(y_test, pred)
    print(f"Accuracy: {round(score*100,2)}% ==== trained in {time.time()-start_time} seconds")
    print(confusion_matrix(y_test, pred))
    print("======================================")

    return confusion_matrix(y_test, pred), score

data = readData(file)

scores_text = []
scores_textemotions = []
overall_conf_text = [[0,0],[0,0]]
overall_conf_textemo = [[0,0],[0,0]]

for ep in range(epoch_number):
    print(f"Current epoch {ep+1}")
    emotion = []
    sampled_data = data.sample(math.floor(len(data)*0.05), random_state=random_state)
    for text in sampled_data["text"]:
        text_obj = NRCLex(text)
        emotion.append(text_obj.top_emotions[0][0])

    sampled_data["emotion"] = emotion
    emotiontext_features = [text + " " + emotion for text, emotion in zip(sampled_data["text"], sampled_data["emotion"])]

    score_and_confusion_text = train_SVM(sampled_data["text"], sampled_data["TruthValue"], "text only")
    scores_text.append(score_and_confusion_text)
    overall_conf_text += score_and_confusion_text[0]

    score_and_confusion_textemotion = train_SVM(emotiontext_features, sampled_data["TruthValue"], "text and emotion")
    scores_textemotions.append(score_and_confusion_textemotion)
    overall_conf_textemo += score_and_confusion_textemotion[0]

    print("=======================================")

overall_conf_text = np.divide(overall_conf_text, len(scores_text))
overall_conf_textemo = np.divide(overall_conf_textemo, len(scores_textemotions))

print(overall_conf_textemo, overall_conf_text)

pickle.dump(scores_text, open("SVM_text.pickle", "wb"))
pickle.dump(scores_textemotions, open("SVM_textemotion.pickle", "wb"))
pickle.dump(overall_conf_text, open("SVM_text_overall.pickle", "wb"))
pickle.dump(overall_conf_textemo, open("SVM_textemotion_overall.pickle", "wb"))


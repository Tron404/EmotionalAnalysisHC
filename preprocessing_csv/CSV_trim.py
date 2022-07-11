import pandas as pd
import numpy as np
import time

path_tweetText = "C:\\Users\\Eduard\\Desktop\\data\\tweets\\hydrated\\COVID_FAKES\\"
path_tweetScore = "C:\\Users\\Eduard\\Desktop\\data\\tweets\\orig_data\\COVID-FAKES-COVID-FAKES-E\\orig\\"
columns = ["id", "lang", "text"]

total_tweets = 0

for i in range(14,32):
    fileCSV_Text = pd.read_csv(path_tweetText + f"COVID_FAKES_{i}.csv", sep=",", index_col=False, skipinitialspace=True, skip_blank_lines=True, dtype=str, usecols=columns)
    fileCSV_Text = fileCSV_Text.dropna()
    fileCSV_Text = fileCSV_Text[fileCSV_Text["lang"] == "en"]
    fileCSV_Text = fileCSV_Text.drop("lang",axis=1)

    try: 
        fileCSV_Score = pd.read_csv(path_tweetScore + f"00{i}.csv", index_col=False, dtype=str, sep="[,\t]")
    except:
        fileCSV_Score = pd.read_csv(path_tweetScore + f"00{i}.csv", index_col=False, dtype=str, sep="\s+")

    print(fileCSV_Score)

    fileCSV_Score = fileCSV_Score.dropna()
    fileCSV_Score = fileCSV_Score[["TweetID", "C_ENS"]]

    fileCSV_Text = fileCSV_Text.sort_values(by=["id"])
    fileCSV_Score = fileCSV_Score.sort_values(by=["TweetID"])

    idx_text = 0
    idx_score = 0
    score_values = []

    start_time = time.time()

    for id in fileCSV_Score["TweetID"].values:
        if idx_text < len(fileCSV_Text) and id == fileCSV_Text.iloc[idx_text,0]:
            aux = fileCSV_Score.iloc[idx_score:]
            score_values.append(aux.loc[aux["TweetID"] == id, "C_ENS"].values[0])
            idx_text += 1
        idx_score += 1

    df_Text_Score = fileCSV_Text.iloc[:idx_text]
    df_Text_Score["TruthValue"] = score_values

    print(df_Text_Score.head())
    print("......")
    print(df_Text_Score.tail())

    df_Text_Score.to_csv(f"Processed\\COVID_FAKES_trimmed_{i}.csv", index=False)
    print(f"File {i} has been saved - there are {len(df_Text_Score)} English tweets")
    print(f"The operation took {time.time()-start_time} seconds")
    print("----------------------------------------------------------------------------------------------")
    total_tweets += len(df_Text_Score)

    print(f"\nIn total, there are {total_tweets} tweets")


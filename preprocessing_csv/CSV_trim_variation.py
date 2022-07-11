import pandas as pd
import time

path_tweetText = "C:\\Users\\Eduard\\Desktop\\data\\tweets\\hydrated\\"
path_tweetOrig = "C:\\Users\\Eduard\\Desktop\\data\\tweets\\orig_data\\"
columns = ["id", "lang", "text"]

total_tweets = 0

file_name = "covid19_disinfo_english_binary"

fileCSV_Text = pd.read_csv(path_tweetText + f"{file_name}_hydrated.csv", sep=",", index_col=False, skipinitialspace=True, skip_blank_lines=True, dtype=str, usecols=columns)
fileCSV_Text = fileCSV_Text.dropna()
fileCSV_Text = fileCSV_Text[fileCSV_Text["lang"] == "en"]
fileCSV_Text = fileCSV_Text.drop("lang",axis=1)

fileCSV_Score = pd.read_csv(path_tweetOrig + f"{file_name}.csv", index_col=False, dtype=str, sep="[,\t]")


fileCSV_Score = fileCSV_Score.dropna()

fileCSV_Text = fileCSV_Text.sort_values(by=["id"])
fileCSV_Score = fileCSV_Score.sort_values(by=["id"])

print(fileCSV_Text.head())

idx_text = 0
idx_score = 0
score_values = []

start_time = time.time()

for id in fileCSV_Score["id"].values:
    if idx_text < len(fileCSV_Text) and id == fileCSV_Text.iloc[idx_text,0]:
        aux = fileCSV_Score.iloc[idx_score:]
        score_values.append(aux.loc[aux["id"] == id, "truth_value"].values[0])
        idx_text += 1
    idx_score += 1

df_Text_Score = fileCSV_Text.iloc[:idx_text]
df_Text_Score["TruthValue"] = score_values

print(df_Text_Score.head())
print("......")
print(df_Text_Score.tail())

df_Text_Score.to_csv(f"Processed\\{file_name}_processed.csv", index=False)
print(f"File has been saved - there are {len(df_Text_Score)} English tweets")
print(f"The operation took {time.time()-start_time} seconds")
print("----------------------------------------------------------------------------------------------")
total_tweets += len(df_Text_Score)

print(f"\nIn total, there are {total_tweets} tweets")


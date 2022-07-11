
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import threading
import time
import spacy
import os

stop_words = spacy.load("en_core_web_sm").Defaults.stop_words

lemmatiser_model = spacy.load("en_core_web_sm", exclude=["parser", "ner", "textcat", "custom"])

def is_number(no):
    try:
        int(no)
        return True
    except:
        return False

def type_token_ratio(frequency):
    tokens = 0
    for word in frequency:
        tokens += word[1]
        types = len(frequency)
        return types/tokens

def zipfian(frequency):
    zip_freq = []
    const = frequency[0][1]

    for i in range(1, len(frequency)+1):
        zip_freq.append(const/i)

    return zip_freq


def print_distribution(frequency_dictionary, zip_distribution, name):
    fig, p = plt.subplots(2)

    # save the frequency values in a separate list
    words = [frequency_dictionary[w][0] for w in range(len(frequency_dictionary))]
    values = [frequency_dictionary[w][1] for w in range(len(frequency_dictionary))]
    
    # plot the 2 distributions
    p[0].scatter(np.log(range(1, len(values)+1)), np.log(values), color="orange", label="Distribution of words in \"%s\"" % "tweets")
    p[0].plot(np.log(range(1, len(zip_distribution)+1)), np.log(zip_distribution), color="black", linewidth=2, label="Zipfian distribution")
    p[0].set_xlabel("Log of Rank", fontsize="x-large")
    p[0].set_ylabel("Log of Frequency", fontsize="x-large")
    p[0].set_title("Frequency distribution", fontsize="x-large")
    p[0].legend(fontsize="x-large")

    number_topwords = 40
    p[1].bar(x=range(len(words[:number_topwords])), height=values[:number_topwords])
    p[1].set_xticks(range(len(words[:number_topwords])), words[:number_topwords], rotation="vertical")

    fig.savefig(f"graphs/{name}.png", dpi=fig.dpi)

    # plt.show()

def remove_empty_strings_spaces(text):
    return [word for word in text if word != "" and word != " "]

def remove_special_cases(text):
    return [word for word in text if (word not in stop_words and ("'" + word) not in stop_words) and not is_number(word) and len(word) >= 2]

def get_lemmas(text):
    return [tk.lemma_ for tk in text]

def dictionary_check(text, dic):
    for word in text:
        if word in dic:
            dic[word] += 1
        else:
            dic[word] = 1

def process_tweets(tweets_text_orig, col_header):
    regex_emoji = r"[\U00000ACB-\U0010ffff]"
    regex_URL = r"(?:https?://)?(?:www.)?(?:[\w\-\.]+)\.[\w]{2,3}(?:/[\w]+){0,}"
    regex_mentions_amp = r"[\w]?[@&][\w]+;?(?:[\w]+)?" # remove mentions, and &amp; constructions
    batch_size = 40
    part = np.array_split(tweets_text_orig, batch_size)

    batch_no = 1
    file_name = "aux.csv"
    file = open(file_name, "a")

    for tweets_text in part:
        start_time = time.time()
        tweets_text["text"] = tweets_text["text"].str.replace(regex_emoji, "", regex=True)
        tweets_text["text"] = tweets_text["text"].str.replace(regex_URL, "", regex=True)
        tweets_text["text"] = tweets_text["text"].str.replace(regex_mentions_amp, "", regex=True)
        tweets_text["text"] = tweets_text["text"].str.lower()

        # lemmatise evertything, which also tokenises the text
        tweets_text["text"] = [lemmatiser_model(text) for text in tweets_text["text"]]
        tweets_text["text"] = [get_lemmas(text) for text in tweets_text["text"]]
        tweets_text["text"] = [remove_special_cases(text) for text in tweets_text["text"]]
        tweets_text["text"] = [remove_empty_strings_spaces(text) for text in tweets_text["text"]]
        tweets_text["text"] = [" ".join(text) for text in tweets_text["text"]]
        tweets_text["text"] = tweets_text["text"].str.lower()

        tweets_text["text"] = tweets_text["text"].str.replace(r"[^ \w]", " ", regex=True)
        tweets_text["text"] = tweets_text["text"].str.replace(r"\n", "", regex=True)
        tweets_text["text"] = tweets_text["text"].str.replace(r" {2,}", " ", regex=True)
        tweets_text["text"] = tweets_text["text"].str.split()
        tweets_text["text"] = [remove_special_cases(text) for text in tweets_text["text"]]
        
        tweets_text.columns = tweets_text.iloc[0]
        tweets_text = tweets_text.iloc[1:]
        tweets_text.to_csv(file, index=False)
        print(f"Batch {batch_no} has finished in {time.time()-start_time} seconds, with {len(tweets_text)} tweeets processed")
        batch_no += 1

    tweets_text_batch_list = pd.read_csv(file_name, sep=",", index_col=False, skip_blank_lines=True, encoding="", dtype=str)
    tweets_text_batch_list.columns = col_header

    tweets_text_batch_list["text"] = tweets_text_batch_list["text"].str.replace(r"[\[\]]", "", regex=True).replace("'", "", regex=True).replace(r" ", "", regex=True).str.split(",")

    file.truncate(0)
    return tweets_text_batch_list

total_tweets = 0
global_dictionary = {}
file_global_name = "global_tweets_features.csv"
file_global_tweets_features = open(file_global_name, "a")
done = os.listdir("features")

for file in os.listdir("data"):
    if file.endswith(".csv") and (file.strip(".csv") + "_features.csv" not in done) and file != file_global_name and file != "aux.csv":
    # if file == "VaxMisinfoData_processed.csv":
        tweets = pd.read_csv("data/" + file, sep=",", index_col=False, skip_blank_lines=True, dtype=str, encoding="") # read the truth value as well!!!
        print(tweets.head())

        start_time = time.time()
        print(f"====There are {len(tweets)} tweets in the original dataset from {file}>")

        concatenated_tweets = process_tweets(tweets, list(tweets.columns))

        # remove duplicate tweets
        concatenated_tweets["string_key"] = concatenated_tweets["text"].apply("".join)
        concatenated_tweets = concatenated_tweets.drop_duplicates("string_key", keep="first", ignore_index=True)

        concatenated_tweets = concatenated_tweets.drop(["string_key"], axis=1)

        word_frequency = {}
        [dictionary_check(text, word_frequency) for text in concatenated_tweets["text"]]
        [dictionary_check(text, global_dictionary) for text in concatenated_tweets["text"]]

        word_frequency = sorted(word_frequency.items(), reverse=True, key=lambda x: x[1])
        zipfian_distribution = zipfian(word_frequency)

        total_tweets += len(concatenated_tweets)

        print(f"====> The process took {time.time()-start_time} seconds, resulting in {len(concatenated_tweets)} duplicate-free tweets\n")
        
        file_no_extension = file.strip(".csv")
        print_distribution(word_frequency, zipfian_distribution, file_no_extension)

        concatenated_tweets.to_csv(f"features/{file_no_extension}_features.csv", index=False, )
        concatenated_tweets.to_csv(file_global_tweets_features, index=False)

global_dictionary = sorted(global_dictionary.items(), reverse=True, key=lambda x: x[1])
global_zipf = zipfian(global_dictionary)
print_distribution(global_dictionary, global_zipf, "all_tweets")
print(f"There are {total_tweets} duplicate-free tweets in total")

global_df = pd.read_csv(file_global_name, sep=",", index_col=False, skip_blank_lines=True, encoding="")
global_df["text"] = global_df["text"].str.replace(r"[\[\]]", "", regex=True).replace("'", "", regex=True).replace(r" ", "", regex=True).str.split(",")

global_df.to_csv("features/all_tweets_features.csv", index=False)
pickle.dump(global_dictionary, open("global_dictionary.pickle", "wb"))


    

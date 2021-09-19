import twitter
import re
import pandas as pd
from nltk.stem.porter import *
import numpy as np
import matplotlib.pyplot as plt

#Remove @username from each tweet
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    return input_txt


test = pd.read_csv("test_tweets_anuFYb8.csv")
train = pd.read_csv("train_E6oV3lV.csv")

#Combine train and test dataset
combi = train.append(test, ignore_index=True, sort=False)

combi['tidy_tweet'] = np.vectorize(remove_pattern)(combi['tweet'], "@[\w]*")

#Remove all numbers and punctuations
combi['tidy_tweet'] = combi['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")

#Remove Short Words
combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 3]))

#Text Normalization (Tokenize tweets)
tokenized_tweet = combi['tidy_tweet'].apply(lambda x: x.split())

#Stemming
stemmer = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])

print(tokenized_tweet[0])

for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
combi['tidy_tweet'] = tokenized_tweet

print(combi.head(7))




### **Code (`sentiment_analysis.py`)**

# -*- coding: utf-8 -*-
"""
Sentiment Analysis with Twitter Datasets
Author: Ebhota Walter Eromosele
"""

# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
from nltk import ngrams
from nltk.tokenize import word_tokenize
from nltk.corpus import opinion_lexicon
import nltk

# Download NLTK resources
nltk.download('punkt')
nltk.download('opinion_lexicon')

# Load datasets
Twitter_US_Airline_Sentiment_Analysis_url = "https://raw.githubusercontent.com/satyajeetkrjha/kaggle-Twitter-US-Airline-Sentiment-/refs/heads/master/Tweets.csv"
twitter_distant_url = "https://media.githubusercontent.com/media/Sultavespa/Sentiment_Analyzer/refs/heads/main/training.1600000.processed.noemoticon.csv"

# Read datasets
Twitter_US_Airline_Sentiment_Analysis = pd.read_csv(Twitter_US_Airline_Sentiment_Analysis_url)
twitter_distant = pd.read_csv(twitter_distant_url, encoding='latin-1')

# Tokenize and extract N-grams
def preprocess(text):
    tokens = word_tokenize(text.lower())
    return tokens

def get_ngrams(text, n=2):
    tokens = preprocess(text)
    return list(ngrams(tokens, n))

# Add bigrams to datasets
Twitter_US_Airline_Sentiment_Analysis['bigrams'] = Twitter_US_Airline_Sentiment_Analysis['text'].apply(lambda x: get_ngrams(x, n=2))
twitter_distant['bigrams'] = twitter_distant.iloc[:, 5].apply(lambda x: get_ngrams(x, n=2))

# Load opinion lexicon
positive_words = set(opinion_lexicon.positive())
negative_words = set(opinion_lexicon.negative())

# Sentiment classification function
def classify_sentiment(text, ngram_model=2):
    pos_count, neg_count = 0, 0
    ngrams_list = get_ngrams(text, n=ngram_model)
    for ngram in ngrams_list:
        if any(word in positive_words for word in ngram):
            pos_count += 1
        if any(word in negative_words for word in ngram):
            neg_count += 1
    if pos_count > neg_count:
        return 'positive'
    elif neg_count > pos_count:
        return 'negative'
    else:
        return 'neutral'

# Classify sentiment
Twitter_US_Airline_Sentiment_Analysis['sentiment'] = Twitter_US_Airline_Sentiment_Analysis['text'].apply(lambda x: classify_sentiment(x))
twitter_distant['sentiment'] = twitter_distant.iloc[:, 5].apply(lambda x: classify_sentiment(x))

# Sentiment distribution
Twitter_US_Airline_Sentiment_Analysis_sentiment_counts = Twitter_US_Airline_Sentiment_Analysis['sentiment'].value_counts()
twitter_distant_sentiment_counts = twitter_distant['sentiment'].value_counts()

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Twitter US Airline Sentiment
Twitter_US_Airline_Sentiment_Analysis_sentiment_counts.plot(kind='bar', ax=axes[0], color=['green', 'red', 'blue'])
axes[0].set_title('Twitter US Airline Sentiment')
axes[0].set_xlabel('Sentiment')
axes[0].set_ylabel('Count')

# Twitter Distant Supervision
twitter_distant_sentiment_counts.plot(kind='bar', ax=axes[1], color=['green', 'red', 'blue'])
axes[1].set_title('Twitter Distant Supervision Sentiment Distribution')
axes[1].set_xlabel('Sentiment')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.show()

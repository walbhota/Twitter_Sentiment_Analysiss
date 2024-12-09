# **Sentiment Analysis with Twitter Datasets** 🐦💬

## Author
Ebhota Walter Eromosele

This project demonstrates sentiment analysis using two popular Twitter datasets: 
- **Twitter US Airline Sentiment Dataset**
- **Twitter Distant Supervision Dataset**

The analysis involves text preprocessing, N-gram extraction, and lexicon-based sentiment classification. The results are visualized to understand the sentiment trends in the datasets.

---

## **Features**

### **Datasets**
1. **[Twitter US Airline Sentiment Dataset](https://raw.githubusercontent.com/satyajeetkrjha/kaggle-Twitter-US-Airline-Sentiment-/refs/heads/master/Tweets.csv)**  
   - Sentiment analysis of tweets about U.S. airlines (Positive, Neutral, Negative).
2. **[Twitter Distant Supervision Dataset](https://media.githubusercontent.com/media/Sultavespa/Sentiment_Analyzer/refs/heads/main/training.1600000.processed.noemoticon.csv)**  
   - Labeled dataset of tweets with noisy labels derived from emoticons.

### **Preprocessing**
- Tokenization of tweet text.
- N-gram extraction (bigrams by default).

### **Sentiment Classification**
- Lexicon-based classification using **NLTK's Opinion Lexicon**:
  - Positive, Negative, or Neutral labels assigned based on word occurrence.

### **Visualization**
- Bar plots to display sentiment distribution for both datasets.

---

## **Setup Instructions**

### **1. Clone the Repository**
```bash
git clone https://github.com/yourusername/sentiment-analysis-twitter.git

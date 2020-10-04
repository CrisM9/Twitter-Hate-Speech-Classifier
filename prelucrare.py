import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS
from nltk import FreqDist
lemma=WordNetLemmatizer()

#my_data=pd.read_csv('train.csv', delimiter=',')

def tweet_column(tweet):
    tweets = " ".join(filter(lambda x: x[0]!='@', tweet.split())) 
    tweets = re.sub('[^a-zA-Z]', ' ', tweets) 
    tweets=tweets.lower() 
    tweets=tweets.split() 
    tweets=[word for word in tweets if not word in set(stopwords.words('english'))] 
    tweets=[lemma.lemmatize(word) for word in tweets] 
    tweets=" ".join(tweets)
    return tweets


def extract_hashtag(tweet):
    tweets=" ".join(filter(lambda x: x[0]=="#", tweet.split())) 
    tweets=re.sub('[^a-zA-Z]', ' ', tweets)
    tweets=tweets.lower()
    tweets=[lemma.lemmatize(word) for word in tweets]
    tweets="".join(tweets)
    return tweets

my_data['processed_text']=my_data.tweet.apply(tweet_column)
my_data['hashtag']=my_data.tweet.apply(extract_hashtag)

total_words=" ".join(my_data.processed_text)
hate_words=" ".join(my_data[my_data['label']==1].processed_text)

wordcloud=WordCloud(height=400, width=400, stopwords=STOPWORDS, background_color="white")
wordcloud=wordcloud.generate(total_words)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

wordcloud=WordCloud(height=400, width=400, stopwords=STOPWORDS, background_color="black")
wordcloud=wordcloud.generate(hate_words)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

hashtag_total=FreqDist(list((" ".join(my_data.hashtag)).split())).most_common(10)
hashtag_hate=FreqDist(list((" ".join(my_data[my_data['label']==1]['hashtag'])).split())).most_common(10)

df_totalhashtag = pd.DataFrame(hashtag_total, columns=['words', 'frequency'])
df_hatehashtag = pd.DataFrame(hashtag_hate, columns=['words', 'frequency'])
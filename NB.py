from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer
import prelucrare
from sklearn.metrics import classification_report

my_data=pd.read_csv('train.csv', delimiter=',')

my_data['processed_text']=my_data.tweet.apply(prelucrare.tweet_column)

labels=my_data['label'].values
messages=my_data['processed_text'].values
shuffle_stratified = StratifiedShuffleSplit(n_splits=1, test_size=0.1)

for train_index, test_index in shuffle_stratified.split(messages, labels):
    msg_train, msg_test = messages[train_index], messages[test_index]
    labels_train, labels_test = labels[train_index], labels[test_index]

CV=CountVectorizer(lowercase=True, analyzer='word') 
CV.fit(msg_train)

X_train=CV.transform(msg_train)
X_test=CV.transform(msg_test)

model=MultinomialNB(alpha=0.01)
model.fit(X_train, labels_train)

predictions = model.predict(X_test)
print(classification_report(labels_test, predictions))
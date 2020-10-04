from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score
import prelucrare

my_data=pd.read_csv('train.csv', delimiter=',')

my_data['processed_text']=my_data.tweet.apply(prelucrare.tweet_column)


vect=TfidfVectorizer(stop_words=stopwords.words('english'),ngram_range=(1,3), min_df=10)
features=vect.fit_transform(my_data.processed_text) #creeare features TFIDF - numerice

X_train, X_test, y_train, y_test = train_test_split(features, my_data.label) 

c_values={'C':[500,1000,3000,7000,1000]}
svc=GridSearchCV(SVC(kernel='rbf',gamma='auto'),param_grid=c_values,scoring='f1',cv=3,n_jobs=-1) 
svc.fit(X_train,y_train) 
print("Parameters {} with F1 score: {:.2f}".format(svc.best_params_ , svc.best_score_))

predict=svc.predict(X_test)
print('F1 SCORE:{}'.format(f1_score(y_test,predict)))
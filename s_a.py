import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle


dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)


nltk.download('stopwords')
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    all_stopwords.remove("wasn't")
    all_stopwords.remove( "isn't")
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)


cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 30)


classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)


pickle.dump(classifier, open('SVMmodel.pkl', 'ab'))
pickle.dump(ps, open('Stemmer.pkl', 'ab'))
pickle.dump(all_stopwords, open('StpWords.pkl', 'ab'))
pickle.dump(cv, open('CountVector.pkl', 'ab'))
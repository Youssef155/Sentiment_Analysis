# use "pyhton app.py" to run the application

from flask import Flask, render_template, request
import pickle
import re
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
model = pickle.load(open('SVMmodel.pkl', 'rb'))
count_vectorizer = pickle.load(open('CountVector.pkl', 'rb'))
stemm = pickle.load(open('Stemmer.pkl', 'rb'))
stp_words = pickle.load(open('StpWords.pkl', 'rb'))

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def classify():
    logger.debug("Processing new review...")
    new_review = request.form['review']
    new_review = re.sub('[^a-zA-Z]', ' ', new_review)
    new_review = new_review.lower()
    new_review = new_review.split()
    new_review = [stemm.stem(word) for word in new_review if not word in set(stp_words)]
    new_review = ' '.join(new_review)
    new_corpus = [new_review]
    new_X_test = count_vectorizer.transform(new_corpus).toarray()
    classifing = model.predict(new_X_test)
    if classifing == 0:
        output = 'Bad Review!'
    elif classifing == 1:
        output = 'Great Review!'
    else:
        logger.warning("Unexpected classification result!") 

    return render_template('index.html', classification = output)
    

if __name__ == "__main__":
    app.run(debug=True)
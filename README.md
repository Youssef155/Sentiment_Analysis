# Sentiment Analysis Application

This project is a **Sentiment Analysis Web Application** built with Flask. It classifies user-provided reviews as either "Great Review!" or "Bad Review!" using a trained Support Vector Machine (SVM) model.

---

## Features

- **User-Friendly Interface**: The application uses an HTML frontend for review submission and result display.
- **Text Preprocessing**: Reviews are cleaned and stemmed before classification.
- **Machine Learning**: Leverages a pre-trained SVM model for sentiment classification.
- **Custom Stopwords and Stemmer**: Implements tailored text-processing tools.

---

## Requirements

### Prerequisites
Ensure you have the following installed:
- Python 3.6+
- Flask
- Required libraries (see below)

### Required Python Libraries
Install the dependencies using the command:
```bash
pip install -r requirements.txt
```

---

## File Structure

```
Sentiment_Analysis/
├── app.py                # Main application file
├── templates/
│   └── index.html       # Frontend for the application
├── SVMmodel.pkl          # Pre-trained SVM model
├── CountVector.pkl       # CountVectorizer for text processing
├── Stemmer.pkl           # Custom stemmer
├── StpWords.pkl          # Custom stopwords
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## How to Run the Application

1. Clone the repository:
   ```bash
   git clone https://github.com/Youssef155/Sentiment_Analysis.git
   cd Sentiment_Analysis
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python app.py
   ```

4. Open your browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

---

## Usage

1. Enter a review in the input box on the webpage.
2. Submit the review.
3. View the classification result: **"Great Review!"** or **"Bad Review!"**.

---

## Technical Details

### Preprocessing Pipeline
1. **Regex Cleaning**: Removes non-alphabetic characters.
2. **Lowercasing**: Converts text to lowercase.
3. **Stopword Removal**: Removes custom-defined stopwords.
4. **Stemming**: Applies a custom stemmer to reduce words to their base forms.

### Machine Learning
- **Model**: SVM (Support Vector Machine)
- **Vectorization**: Utilizes CountVectorizer for transforming text into numerical format.
- **Pickle**: Saves the model, vectorizer, and other pre-processing tools for reuse.

---

## Logging
The application logs important events, such as:
- **Debugging Information**: Tracks preprocessing and classification steps.
- **Warnings**: Alerts for unexpected classification outputs.

---

## Acknowledgments
- **Libraries Used**: Flask, scikit-learn, and more.
- **Developer**: Youssef155


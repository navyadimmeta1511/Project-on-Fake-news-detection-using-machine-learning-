from flask import Flask, request, render_template, redirect, url_for
import joblib
import numpy as np

# Load the model and vectorizer
model = joblib.load('fake_tweet_classifier.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html') 

@app.route('/index')
def index():
    return render_template('index.html')  

@app.route('/about')
def about():
    return render_template('about.html') 

@app.route('/contact')
def contact():
    return render_template('contact.html') 


@app.route('/performance')
def performance():
    return render_template('performance.html') 


# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the tweet from the form
    tweet = request.form['tweet']

    # Preprocess the tweet
    cleaned_tweet = clean_text(tweet)  # Use your preprocessing function
    tweet_vector = vectorizer.transform([cleaned_tweet]).toarray()

    # Make a prediction
    prediction = model.predict(tweet_vector)
    result = "Real" if prediction[0] == 1 else "Fake"

    # Return the result to the HTML template
    return render_template('index.html', prediction_text=f'The tweet is {result}')

# Text cleaning function (same as before)
def clean_text(text):
    import re
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)



VALID_USERNAME = "user"
VALID_PASSWORD = "123456"

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/authenticate', methods=['POST'])   
def authenticate():
    username = request.form.get('username')
    password = request.form.get('password')

    if username == VALID_USERNAME and password == VALID_PASSWORD:
        return redirect(url_for('index'))
    else:
        return render_template('login.html', error="Invalid Username or Password!")


if __name__ == '__main__':
    app.run(debug=True)











































































###############################################################################

# from flask import Flask, request, render_template
# import joblib

# app = Flask(__name__)

# # Load your model and vectorizer
# model = joblib.load('fake_tweet_classifier.pkl')
# vectorizer = joblib.load('tfidf_vectorizer.pkl')

# # Home route
# @app.route('/')
# def home():
#     return render_template('home.html')  # Render the home page

# # Prediction route (accepts both GET and POST)
# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
#     if request.method == 'POST':
#         # Handle POST request (form submission)
#         tweet = request.form['tweet']
#         cleaned_tweet = clean_text(tweet)
#         tweet_vector = vectorizer.transform([cleaned_tweet]).toarray()
#         prediction = model.predict(tweet_vector)
#         result = "Real" if prediction[0] == 1 else "Fake"
#         return render_template('index.html', prediction_text=f'The tweet is {result}')
#     else:
#         # Handle GET request (display a message)
#         return "Please submit a tweet using the form on the home page."

# # Text cleaning function
# def clean_text(text):
#     import re
#     from nltk.corpus import stopwords
#     from nltk.tokenize import word_tokenize

#     # Remove special characters and numbers
#     text = re.sub(r'[^a-zA-Z\s]', '', text)
#     # Convert to lowercase
#     text = text.lower()
#     # Remove stopwords
#     stop_words = set(stopwords.words('english'))
#     tokens = word_tokenize(text)
#     tokens = [word for word in tokens if word not in stop_words]
#     return ' '.join(tokens)

# # Run the Flask app
# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from authlib.integrations.flask_client import OAuth
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import joblib
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')

oauth = OAuth(app)
oauth.register(
    'auth0',
    client_id=os.getenv('AUTH0_CLIENT_ID'),
    client_secret=os.getenv('AUTH0_CLIENT_SECRET'),
    authorize_url=f'https://{os.getenv("AUTH0_DOMAIN")}/authorize',
    access_token_url=f'https://{os.getenv("AUTH0_DOMAIN")}/oauth/token',
    authorize_params=None,
    client_kwargs={
        'scope': 'openid profile email',
    },
)

# Load the trained model and TF-IDF vectorizer
model = joblib.load('ml_model/svm_model.pkl')
tfidf = joblib.load('ml_model/tfidf_vectorizer.pkl')

# Text preprocessing function
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert text to lowercase
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = ' '.join([word for word in word_tokens if word not in stop_words])  # Remove stopwords
    return filtered_text

@app.route('/login')
def login():
    redirect_uri = url_for('auth0_authorize', _external=True)
    return oauth.auth0.authorize_redirect(redirect_uri)

@app.route('/logout')
def logout():
    # Clear the session and log the user out from Auth0
    session.clear()
    return redirect(oauth.auth0.end_session_url)

@app.route('/auth0/callback')
def auth0_authorize():
    token = oauth.auth0.authorize_access_token()
    userinfo = oauth.auth0.parse_id_token(token)
    # Store userinfo or handle the user data as needed
    return redirect('/')

@app.route('/register')  # New route for registration page
def register():
    redirect_uri = url_for('auth0_authorize', _external=True)
    return oauth.auth0.authorize_redirect(redirect_uri, audience='https://dev-iwzuwxeyjtsg07au.us.auth0.com/api/v2/users')

@app.route('/', methods=['GET', 'POST'])
def predict_sentiment():
    if request.method == 'POST':
        user_review = request.json.get('user_review', '')
        cleaned_review = preprocess_text(user_review)
        input_tfidf = tfidf.transform([cleaned_review])
        prediction = model.predict(input_tfidf)
        return jsonify({'prediction': prediction[0]})
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

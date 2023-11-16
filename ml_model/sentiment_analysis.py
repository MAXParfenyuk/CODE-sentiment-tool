# Import necessary libraries
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# Load the dataset
dataset = pd.read_csv('IMDB Dataset.csv')

# Explore the dataset
print(dataset.head())
print(dataset.columns)
print(dataset.shape)
print(dataset.isnull().sum())

nltk.download('punkt')
nltk.download('stopwords')
# Handle missing values
dataset.dropna(inplace=True)

# Text preprocessing
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert text to lowercase
    text = ' '.join([word for word in word_tokenize(text) if word not in stopwords.words('english')])  # Remove stopwords
    return text

dataset['cleaned_reviews'] = dataset['review'].apply(preprocess_text)

# Split the dataset
X = dataset['cleaned_reviews']
y = dataset['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature extraction using TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)

# Train a Sentiment Analysis Model
model = SVC()
model.fit(X_train_tfidf, y_train)

# Evaluate the model
X_test_tfidf = tfidf.transform(X_test)
accuracy = model.score(X_test_tfidf, y_test)
print(f"Model Accuracy: {accuracy}")

# Tuning the model
tuned_model = SVC(kernel='linear', C=1.0)
tuned_model.fit(X_train_tfidf, y_train)

# Evaluate the tuned model
accuracy_tuned = tuned_model.score(X_test_tfidf, y_test)
print(f"Tuned Model Accuracy: {accuracy_tuned}")

# Predictions
y_pred = tuned_model.predict(X_test_tfidf)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

import joblib

# Save the trained models
joblib.dump(model, 'svm_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

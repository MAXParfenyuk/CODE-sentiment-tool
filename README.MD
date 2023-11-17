# Sentiment Analysis using Support Vector Machines (SVM)

This project focuses on sentiment analysis utilizing Support Vector Machines (SVM) to classify movie reviews' sentiments (positive or negative). 

## AI Problem Description

### Problem Name: Sentiment Analysis for Movie Reviews

#### Problem Description: 
The problem aims to classify movie reviews into positive or negative sentiments based on their textual content. The dataset consists of movie reviews from IMDB.

#### Area of AI: 
Machine Learning

#### Applied Algorithms: 
- Text Preprocessing: Punctuation removal, converting text to lowercase, stop words removal.
- Feature Extraction: TF-IDF (Term Frequency-Inverse Document Frequency).
- Machine Learning Algorithm: Support Vector Machines (SVM).

#### Results: 
The model achieved a satisfactory accuracy of [90]% on the test set. By tuning the SVM model with a linear kernel and C=1.0, the accuracy improved to [89]% on the same test set.

## Code Location
The AI model and learning code can be found in the following file:
- [sentiment_analysis.py](CODE-sentiment-tool/ml_model/sentiment_analysis.py)
The dataset can be found in the following file:
- [IMDB Dataset.csv](CODE-sentiment-tool/ml_model/IMDB Dataset.csv)
---
## IMDB Dataset

The project utilizes the IMDB Dataset, a CSV file containing movie reviews used for sentiment analysis. The dataset contains two columns:
- `review`: Textual content of the movie reviews.
- `sentiment`: Label indicating the sentiment of the review (positive/negative).

### Dataset Details
- **File Name:** IMDB Dataset.csv
- **File Location:** [path/to/your/IMDB Dataset.csv]
- **Structure:** 
  - Column 1: `review` - Textual content of the reviews.
  - Column 2: `sentiment` - Label indicating the sentiment (positive/negative).
- **Usage:** The dataset is used to train and test the sentiment analysis machine learning model.

### Dataset Example
Here's a snippet of the dataset:
	review	sentiment
1	A very good movie. I enjoyed it a lot.	positive
2	The acting was terrible. Not worth watching.	negative
3	Brilliant film! Highly recommended.	positive
...	...	...

## How to Set Up the Development Environment

### Prerequisites
- Python 3.x
- Libraries: pandas, nltk, matplotlib, seaborn, scikit-learn

### Steps to Run the AI Model Locally
1. Clone this repository:
   ```bash
   git clone https://github.com/MAXParfenyuk/CODE-sentiment-tool.git
2. Install the required libraries:
    pip install pandas nltk matplotlib seaborn scikit-learn
3. navigate to the project directory
4. Execute the Ai model code:
    python sentiment_analysis.py


### Training and Saving the Model
The trained SVM model and TF-IDF vectorizer can be found as:

svm_model.pkl
tfidf_vectorizer.pkl


# Sentiment Analysis Web App

This Flask-based web application provides sentiment analysis for movie reviews using a trained machine learning model (SVM). It uses OAuth integration with Auth0 for user authentication.

## Getting Started

Follow these steps to set up and run the application locally.

### Prerequisites

- Python 3.x
- Pip (Python package installer)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/MAXParfenyuk/CODE-sentiment-tool.git
2. Navigate to the project directory:
    cd your-repository
3.  pip install -r requirements.txt
    pip install pandas Flask authlib nltk scikit-learn matplotlib seaborn

4. run the application localy:
    python app.py
5. Check web origins:
    https://sentiment-ai-787fe96029f3.herokuapp.com

### API Endpoints (doesnt work yet)
/login: Initiates the Auth0 login process.
/logout: Logs the user out from Auth0.
/auth0/callback: Callback URL after successful Auth0 authentication.
/register: Route for user registration.
/: Main route for sentiment analysis prediction (requires authentication).

### Note
Ensure the ml_model/svm_model.pkl and ml_model/tfidf_vectorizer.pkl files containing the trained model and TF-IDF vectorizer are available.

### Built With
Flask - Web framework
Authlib - OAuth integration
NLTK - Natural Language Toolkit
Scikit-learn - Machine learning library
Pandas - Data manipulation library
Matplotlib and Seaborn - Data visualization libraries
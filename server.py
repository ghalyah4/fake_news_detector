from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import re
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and vectorizer
with open("model.pkl copy 3", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl copy 3", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

app = Flask(__name__)  # Flask application initialization
CORS(app)  # Enable CORS to prevent cross-origin issues

@app.route('/')
def index():
    return render_template('index.html')

def preprocess_text(text):
    """Preprocess input text similarly to how the model was trained."""
    text = text.lower()
    text = re.sub(r'\[\d*\]', '', text)  # Remove references like [1]
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'https?:\/\/\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()  # Ensure JSON parsing
        text = data.get('text', '').strip()  # Extract 'text' safely

        if not text:  # Handle empty input
            return jsonify({'error': 'No text provided'}), 400

        # Preprocess and vectorize input text
        processed_text = preprocess_text(text)
        vectorized_text = vectorizer.transform([processed_text])

        # Predict using the trained model
        prediction = model.predict(vectorized_text)[0]
        probability = model.predict_proba(vectorized_text)[0][prediction] * 100  # Fix probability retrieval

        result = "Fake News" if prediction == 0 else "Not A Fake News"

        return jsonify({
            'prediction': result,
            'probability': round(probability, 2)  # Rounded probability %
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import requests
import joblib
import datetime as dt
import yfinance as yf
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# --- Flask & Model Setup ---
app = Flask(__name__)

# Load pre-trained model and scaler
model = load_model('tsla_model.h5')
scaler = joblib.load('scaler.pkl')

# --- FinBERT Setup for real-time sentiment ---
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model_finbert = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def get_sentiment_score(text):
    """
    (Same function as in collect_data.py)
    """
    if not text:
        return 0.0
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model_finbert(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    positive_prob = probabilities[:, 0].item()
    negative_prob = probabilities[:, 1].item()
    return positive_prob - negative_prob

def predict_next_day_price(symbol, marketaux_api_key, look_back=60):
    """
    Fetches real-time data, prepares it, and returns a prediction.
    """
    # Fetch latest price data for the last 'look_back' days
    end_date = dt.datetime.now()
    # Get a few extra days just in case to ensure we have enough for the look_back window
    start_date = end_date - dt.timedelta(days=look_back + 10) 
    
    price_df = yf.download(symbol, start=start_date, end=end_date)
    
    # Get today's sentiment score from the latest news
    params = {
        'api_token': marketaux_api_key,
        'symbols': symbol
    }
    try:
        response = requests.get('https://api.marketaux.com/v1/news/all', params=params)
        data = response.json()
        headlines = [article['title'] + ' ' + article.get('description', '') for article in data.get('data', [])]
        daily_sentiment_scores = [get_sentiment_score(text) for text in headlines]
        latest_sentiment = sum(daily_sentiment_scores) / len(daily_sentiment_scores) if daily_sentiment_scores else 0.0
    except Exception as e:
        print(f"Error fetching real-time sentiment: {e}")
        latest_sentiment = 0.0
    
    # Add sentiment column to the dataframe for a full feature set
    price_df['Sentiment'] = latest_sentiment
    
    # Get the last 'look_back' days of data
    last_60_days = price_df[['Close', 'Sentiment']].tail(look_back).values
    
    # Scale the data and make the prediction
    last_60_days_scaled = scaler.transform(last_60_days)
    X_test = np.array([last_60_days_scaled])
    
    predicted_price_scaled = model.predict(X_test, verbose=0)
    
    # Inverse transform to get the actual price
    dummy_array = np.zeros((1, 2))
    dummy_array[0, 0] = predicted_price_scaled
    predicted_price = scaler.inverse_transform(dummy_array)[0, 0]
    
    return predicted_price

# Main page route
@app.route('/')
def home():
    return render_template('index.html')

# API endpoint for prediction
@app.route('/predict', methods=['GET'])
def predict():
    try:
        MARKETAUX_API_KEY = "YOUR_MARKETAUX_API_KEY"
        predicted_price = predict_next_day_price('TSLA', MARKETAUX_API_KEY)
        
        # Get the current date and time
        prediction_date = dt.datetime.now().strftime("%B %d, %Y %I:%M %p")
        
        return jsonify({
            'symbol': 'TSLA',
            'prediction': f'${predicted_price:.2f}',
            'message': "Prediction for next trading day's closing price.",
            'date': prediction_date
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
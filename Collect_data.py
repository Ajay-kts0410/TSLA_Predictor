import pandas as pd
import yfinance as yf
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import datetime as dt
import os

# --- FinBERT Setup ---
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model_finbert = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def get_sentiment_score(text):
    if not text:
        return 0.0
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model_finbert(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    positive_prob = probabilities[:, 0].item()
    negative_prob = probabilities[:, 1].item()
    sentiment_score = positive_prob - negative_prob
    return sentiment_score

def create_historical_dataset(symbol, marketaux_api_key, start_date, end_date):
    
    file_path = 'TSLA_with_sentiment.csv'
    
    # Check for existing data to resume collection
    if os.path.exists(file_path):
        print("Existing data file found. Resuming data collection.")
        existing_df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        # Handle the MultiIndex issue on load
        if existing_df.index.nlevels > 1:
            existing_df = existing_df.reset_index(level=1, drop=True)
        
        start_collect_date = existing_df.index.max().date() + dt.timedelta(days=1)
    else:
        print("No existing data file found. Starting from scratch.")
        existing_df = pd.DataFrame() # Start with an empty DataFrame
        start_collect_date = dt.datetime.strptime(start_date, '%Y-%m-%d').date()

    print("Fetching and processing news data...")
    sentiment_data = {}
    
    current_date = start_collect_date
    
    # Wrap the loop in a try-except block to handle API limits
    try:
        while current_date <= dt.datetime.strptime(end_date, '%Y-%m-%d').date():
            params = {
                'api_token': marketaux_api_key,
                'symbols': symbol,
                'published_on': current_date.isoformat()
            }
            
            response = requests.get('https://api.marketaux.com/v1/news/all', params=params)
            data = response.json()
            
            if 'data' in data and data['data']:
                headlines = [article['title'] + ' ' + article.get('description', '') for article in data['data']]
                daily_sentiment_scores = [get_sentiment_score(text) for text in headlines]
                avg_sentiment = sum(daily_sentiment_scores) / len(daily_sentiment_scores) if daily_sentiment_scores else 0.0
            else:
                avg_sentiment = 0.0
            
            sentiment_data[current_date] = avg_sentiment
            print(f"Processed news for {current_date}: Avg Sentiment = {avg_sentiment:.4f}")

            current_date += dt.timedelta(days=1)
            
    except Exception as e:
        print(f"Error encountered: {e}")
    
    # --- SAVE LOGIC ---
    print("Saving collected data...")
    new_sentiment_df = pd.DataFrame.from_dict(sentiment_data, orient='index', columns=['Sentiment'])
    new_sentiment_df.index.name = 'Date'

    if not new_sentiment_df.empty:
        price_start_date = new_sentiment_df.index.min()
        price_end_date = new_sentiment_df.index.max()
        
        price_df = yf.download(symbol, start=price_start_date, end=price_end_date)
        
        # This is the new fix: explicitly flatten the columns after downloading
        price_df.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in price_df.columns]

        # Merge price and sentiment data
        merged_df = price_df.join(new_sentiment_df, how='left')
    else:
        print("No new data collected in this session. Exiting.")
        return

    # If an existing file was found, concatenate old and new data
    if not existing_df.empty:
        merged_df = pd.concat([existing_df, merged_df])
        merged_df = merged_df[~merged_df.index.duplicated(keep='last')]
    
    merged_df.to_csv(file_path)
    print("Combined dataset saved to TSLA_with_sentiment.csv")

if __name__ == "__main__":
    MARKETAUX_API_KEY = "TPwh60C6HtSuR7Vi4ZzLAZAy4ro3DntnLcklkh0x"
    START_DATE = "2014-09-12" 
    END_DATE = dt.date.today().isoformat()
    create_historical_dataset('TSLA', MARKETAUX_API_KEY, START_DATE, END_DATE)
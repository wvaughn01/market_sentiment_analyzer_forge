# Import packages
import os
import json
import requests
from datetime import datetime, timedelta
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from newsapi import NewsApiClient
import streamlit as st

# Only import API clients if we're not in mock mode
USE_MOCK = os.environ.get('USE_MOCK_SCRAPE', '0') == '1'

class FinancialNewsAggregator:
    def __init__(self, alpha_vantage_key=None, newsapi_key=None, use_mock=None):
        """
        Initialize with API keys. If keys are not provided, will attempt to read from environment variables:
        ALPHA_VANTAGE_KEY, NEWS_API_KEY
        """
        self.use_mock = use_mock if use_mock is not None else os.environ.get('USE_MOCK_SCRAPE', '0') == '1'
        
        if not self.use_mock:
            self.alpha_vantage_key = alpha_vantage_key or st.secrets.get("ALPHA_VANTAGE_KEY")
            self.newsapi_key = newsapi_key or st.secrets.get("NEWS_API_KEY")

            
            # Initialize NewsAPI client if key is available
            if self.newsapi_key:
                self.newsapi_client = NewsApiClient(api_key=self.newsapi_key)
            
        self.analyzer = SentimentIntensityAnalyzer()

    def get_alpha_vantage_news(self, ticker):
        if not self.alpha_vantage_key:
            return pd.DataFrame()
        try:
            url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={self.alpha_vantage_key}"
            response = requests.get(url)
            data = response.json()
            if 'feed' not in data:
                print(f"No news data found: {data.get('Note', 'Unknown error')}")
                return pd.DataFrame()
            news_items = []
            for item in data['feed']:
                content = f"{item.get('title', '')} {item.get('summary', '')}"
                # ✅ FIXED: Alpha Vantage date parsing
                raw_date = item.get('time_published', '')
                parsed_date = datetime.strptime(raw_date, "%Y%m%dT%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
                news_items.append({
                    'date': parsed_date,
                    'content': content,
                    'source': item.get('source', ''),
                    'url': item.get('url', ''),
                    'sentiment': item.get('overall_sentiment_score') or self.analyze_sentiment(content)
                })
            return pd.DataFrame(news_items)
        except Exception as e:
            print(f"Error fetching Alpha Vantage news: {str(e)}")
            return pd.DataFrame()


    def get_newsapi_articles(self, ticker, days_back=7):
        if not self.newsapi_key:
            return pd.DataFrame()
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            response = self.newsapi_client.get_everything(
                q=ticker,
                from_param=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d'),
                language='en',
                sort_by='relevancy'
            )
            if not response or 'articles' not in response:
                return pd.DataFrame()
            articles = []
            for article in response['articles']:
                content = f"{article['title']} {article['description']}"
                # ✅ Consistent NewsAPI date parsing
                parsed_date = datetime.strptime(article['publishedAt'], "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d %H:%M:%S")
                articles.append({
                    'date': parsed_date,
                    'content': content,
                    'source': article['source']['name'],
                    'url': article['url'],
                    'sentiment': self.analyze_sentiment(content)
                })
            return pd.DataFrame(articles)
        except Exception as e:
            print(f"Error fetching NewsAPI articles: {str(e)}")
            return pd.DataFrame()



    def get_mock_news(self, ticker, start_date, end_date):
        """Returns mock news data from sample_news.json"""
        try:
            news_items = []
            sample_path = os.path.join(os.path.dirname(__file__), 'sample_news.json')
            if not os.path.exists(sample_path):
                print(f"Warning: Mock data file {sample_path} not found")
                return pd.DataFrame()
                
            with open(sample_path, 'r') as f:
                for line in f:
                    item = json.loads(line.strip())
                    item_date = datetime.strptime(item['date'], '%Y-%m-%dT%H:%M:%S%z')
                    item['date'] = item_date.strftime('%Y-%m-%d %H:%M:%S')
                    news_items.append(item)
            
            df = pd.DataFrame(news_items)
            
            # Filter by date range
            df['date'] = pd.to_datetime(df['date'])
            mask = (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))
            df = df[mask]
            
            # Filter by ticker if specified
            if ticker:
                df = df[df['content'].str.contains(ticker, case=False)]
                
            return df
        except Exception as e:
            print(f"Error loading mock news: {str(e)}")
            return pd.DataFrame()

    def get_sentiment_data(self, ticker, start_date, end_date):
        """
        Get news and sentiment data from all available sources for a given ticker between start and end dates.
        Returns a pandas DataFrame with normalized data from all sources.
        """
        if self.use_mock:
            df = self.get_mock_news(ticker, start_date, end_date)
            if not df.empty:
                print(f"[DEV MODE] Found {len(df)} news items between {start_date} and {end_date}")
            return df
        
        # Get news from all available sources
        dfs = []
        
        # Alpha Vantage news
        av_news = self.get_alpha_vantage_news(ticker)
        if not av_news.empty:
            av_news['source_type'] = 'alpha_vantage'
            dfs.append(av_news)
        
        # NewsAPI articles
        news_articles = self.get_newsapi_articles(ticker)
        if not news_articles.empty:
            news_articles['source_type'] = 'newsapi'
            dfs.append(news_articles)
        
        # Combine all sources
        if not dfs:
            return pd.DataFrame()
            
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df['date'] = pd.to_datetime(combined_df['date'])
        
        # Filter by date range
        mask = (combined_df['date'] >= pd.to_datetime(start_date)) & (combined_df['date'] <= pd.to_datetime(end_date))
        return combined_df[mask]

    def analyze_sentiment(self, text):
        """Analyze sentiment of text using VADER"""
        return self.analyzer.polarity_scores(str(text))['compound']

    def get_avg_sentiment(self, df):
        """Calculate average sentiment from a DataFrame of news items"""
        if df.empty:
            return 0.0
        if 'sentiment' not in df.columns:
            df['sentiment'] = df['content'].apply(self.analyze_sentiment)
        return df['sentiment'].mean()

# TextBlob Sentiment Labeler 
from textblob import TextBlob

def analyze_sentiment_label(text):
    """
    Analyze sentiment of a given text using TextBlob.

    Returns:
        str: Sentiment label with emoji.
    """
    blob = TextBlob(str(text))
    polarity = blob.sentiment.polarity

    if polarity > 0.1:
        return "😊 Positive"
    elif polarity < -0.1:
        return "😠 Negative"
    else:
        return "😐 Neutral"
    
# Reusable VADER function 
vader = SentimentIntensityAnalyzer()

def analyze_sentiment_vader(text):
    """
    Analyze sentiment of a given text using VADER.
    Returns a compound score between -1.0 and 1.0.
    """
    return vader.polarity_scores(str(text))['compound']


import requests
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize VADER (assumes NLTK data is properly set up)
sid = SentimentIntensityAnalyzer()

class SentimentAnalyzer:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def get_stock_news(self, ticker):

     try:
        url = f"https://finance.yahoo.com/quote/{ticker}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        news_section = soup.find("div", {"id": "quoteNewsStream-0-Stream"})
        if not news_section:
            return None
            
        headlines = [h.text.strip() for h in news_section.find_all("h3")][:5]
        return headlines or None
        
     except Exception as e:
        logger.error(f"Error fetching news for {ticker}: {str(e)}")
        return None

    def analyze(self, ticker):
        """Analyze sentiment"""
        headlines = self.get_stock_news(ticker)
        if not headlines:
            return None
            
        scores = [sid.polarity_scores(h)["compound"] for h in headlines]
        return round(sum(scores) / len(scores), 2)

# Global analyzer instance
analyzer = SentimentAnalyzer()

def analyze_sentiment(ticker):
    """Public interface for sentiment analysis"""
    return analyzer.analyze(ticker)
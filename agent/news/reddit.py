from __future__ import annotations

from typing import List
import requests
import time
import random

from ..logging_config import get_logger

logger = get_logger(__name__)

def fetch_reddit_posts(subreddits: List[str], limit_per_subreddit: int = 10) -> List[str]:
    """
    Fetch posts from Reddit subreddits for sentiment analysis.
    
    Args:
        subreddits: List of subreddit names (without r/)
        limit_per_subreddit: Number of posts to fetch per subreddit
        
    Returns:
        List of post titles and selftext for sentiment analysis
    """
    logger.info(f"Fetching Reddit posts from {len(subreddits)} subreddits, limit per subreddit: {limit_per_subreddit}")
    
    texts: List[str] = []
    headers = {
        'User-Agent': 'TradingBot/1.0 (Educational Purpose)'
    }
    
    for i, subreddit in enumerate(subreddits):
        logger.debug(f"Processing subreddit {i+1}/{len(subreddits)}: r/{subreddit}")
        
        try:
            # Fetch hot posts from subreddit
            url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={limit_per_subreddit}"
            logger.debug(f"Making request to Reddit API: r/{subreddit}")
            
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            posts = data.get('data', {}).get('children', [])
            
            if not posts:
                logger.warning(f"No posts found in r/{subreddit}")
                continue
            
            logger.debug(f"Found {len(posts)} posts in r/{subreddit}")
            
            subreddit_texts = []
            for j, post in enumerate(posts):
                post_data = post.get('data', {})
                title = post_data.get('title', '')
                selftext = post_data.get('selftext', '')
                
                # Combine title and selftext for better sentiment analysis
                if selftext and len(selftext) > 50:  # Only include substantial selftext
                    text = f"{title} - {selftext[:200]}..."  # Limit selftext length
                else:
                    text = title
                
                if text and len(text.strip()) > 10:  # Filter out very short posts
                    subreddit_texts.append(text.strip())
                    logger.debug(f"Post {j+1}: {title[:50]}...")
                else:
                    logger.debug(f"Post {j+1}: skipped (too short or empty)")
            
            logger.info(f"Subreddit r/{subreddit}: processed {len(subreddit_texts)} valid posts")
            texts.extend(subreddit_texts)
            
            # Add small delay to be respectful to Reddit API
            time.sleep(0.5)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch from r/{subreddit}: {e}")
            continue
        except Exception as e:
            logger.error(f"Unexpected error fetching from r/{subreddit}: {e}")
            continue
    
    logger.info(f"Total Reddit posts collected: {len(texts)}")
    if texts:
        logger.debug(f"Sample Reddit posts: {texts[:2]}")
    
    return texts


def fetch_crypto_reddit_posts(limit_per_subreddit: int = 8) -> List[str]:
    """
    Fetch posts from popular crypto-related subreddits.
    
    Args:
        limit_per_subreddit: Number of posts to fetch per subreddit
        
    Returns:
        List of post texts for sentiment analysis
    """
    crypto_subreddits = [
        'cryptocurrency',  # General crypto discussion
        'bitcoin',         # Bitcoin focused
        'ethereum',        # Ethereum focused
        'cryptomarkets',   # Market discussions
        'cryptocurrencytrading',  # Trading focused
        'altcoin'          # Altcoin discussions
    ]
    
    return fetch_reddit_posts(crypto_subreddits, limit_per_subreddit)


def fetch_stock_reddit_posts(limit_per_subreddit: int = 8) -> List[str]:
    """
    Fetch posts from popular stock-related subreddits.
    
    Args:
        limit_per_subreddit: Number of posts to fetch per subreddit
        
    Returns:
        List of post texts for sentiment analysis
    """
    stock_subreddits = [
        'stocks',          # General stock discussion
        'investing',       # Investment discussions
        'SecurityAnalysis', # Fundamental analysis
        'ValueInvesting',  # Value investing
        'StockMarket',     # Market discussions
        'wallstreetbets'   # High activity trading discussions
    ]
    
    return fetch_reddit_posts(stock_subreddits, limit_per_subreddit)

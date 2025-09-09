from __future__ import annotations

from typing import List, Optional
import requests
import time
import random
from datetime import datetime, timedelta, timezone

from ..logging_config import get_logger

logger = get_logger(__name__)

def fetch_reddit_posts(subreddits: List[str], limit_per_subreddit: int = 10, symbol: Optional[str] = None, hours_back: int = 6) -> List[str]:
    """
    Fetch posts from Reddit subreddits for sentiment analysis.
    
    Args:
        subreddits: List of subreddit names (without r/)
        limit_per_subreddit: Number of posts to fetch per subreddit
        symbol: Optional symbol to filter posts (e.g., "BTC/USDT", "ETH/USDT")
        hours_back: Only fetch posts from last N hours (default: 6)
        
    Returns:
        List of post titles and selftext for sentiment analysis
    """
    logger.info(f"Fetching Reddit posts from {len(subreddits)} subreddits, limit per subreddit: {limit_per_subreddit}")
    if symbol:
        logger.info(f"Filtering Reddit posts for symbol: {symbol}")
    logger.info(f"Time filter: Only posts from last {hours_back} hours")
    
    # Calculate cutoff time for recent posts (using UTC)
    cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)
    cutoff_timestamp = cutoff_time.timestamp()
    logger.info(f"Cutoff time (UTC): {cutoff_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    
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
                    # Filter by time (only recent posts)
                    post_time = post_data.get('created_utc', 0)
                    if post_time:
                        try:
                            # Reddit timestamps are already in UTC
                            post_datetime = datetime.fromtimestamp(post_time, tz=timezone.utc)
                            
                            if post_datetime < cutoff_time:
                                logger.debug(f"Post {j+1}: filtered out (too old, {post_datetime.strftime('%Y-%m-%d %H:%M:%S UTC')}): {title[:50]}...")
                                continue
                            else:
                                logger.debug(f"Post {j+1}: time OK ({post_datetime.strftime('%Y-%m-%d %H:%M:%S UTC')}): {title[:50]}...")
                        except (ValueError, TypeError) as e:
                            logger.debug(f"Post {j+1}: time parsing error, including anyway: {e}")
                            # If we can't parse the time, include the post to be safe
                    else:
                        logger.debug(f"Post {j+1}: no time info, including anyway: {title[:50]}...")
                    
                    # Filter by symbol if provided
                    if symbol and not _is_relevant_to_symbol(text, symbol):
                        logger.debug(f"Post {j+1}: filtered out (not relevant to {symbol}): {title[:50]}...")
                        continue
                    
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


def _is_relevant_to_symbol(text: str, symbol: str) -> bool:
    """
    Check if a Reddit post is relevant to the given symbol.
    Reuses the same logic as RSS filtering for consistency.
    
    Args:
        text: The post title and content text
        symbol: The trading symbol (e.g., "BTC/USDT", "ETH/USDT", "AAPL")
    
    Returns:
        True if the text is relevant to the symbol, False otherwise
    """
    # Import the RSS filtering function to reuse the same logic
    from .rss import _is_relevant_to_symbol as rss_is_relevant
    return rss_is_relevant(text, symbol)

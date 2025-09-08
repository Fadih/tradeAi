"""
Sentiment analysis utilities for interpreting sentiment scores
"""

from typing import Tuple, Dict, Any
from .config import get_config

def interpret_sentiment_score(score: float, config=None) -> Dict[str, Any]:
    """
    Interpret a sentiment score using configuration thresholds.
    
    Args:
        score: Sentiment score from -1 to 1
        config: Optional config object, will load if not provided
        
    Returns:
        Dictionary with sentiment interpretation
    """
    if config is None:
        config = get_config()
    
    # Get thresholds from configuration
    positive_threshold = config.sentiment_analysis.positive_threshold
    negative_threshold = config.sentiment_analysis.negative_threshold
    neutral_range = config.sentiment_analysis.neutral_range
    
    # Determine sentiment category
    if score >= positive_threshold:
        category = "positive"
        label = "Bullish"
        color = "success"
        description = f"Strong positive sentiment (≥{positive_threshold})"
    elif score <= negative_threshold:
        category = "negative"
        label = "Bearish"
        color = "danger"
        description = f"Strong negative sentiment (≤{negative_threshold})"
    elif neutral_range[0] <= score <= neutral_range[1]:
        category = "neutral"
        label = "Neutral"
        color = "secondary"
        description = f"Neutral sentiment ({neutral_range[0]} to {neutral_range[1]})"
    else:
        # Score is outside neutral range but not reaching positive/negative thresholds
        if score > 0:
            category = "weakly_positive"
            label = "Slightly Bullish"
            color = "info"
            description = f"Weak positive sentiment ({neutral_range[1]} to {positive_threshold})"
        else:
            category = "weakly_negative"
            label = "Slightly Bearish"
            color = "warning"
            description = f"Weak negative sentiment ({negative_threshold} to {neutral_range[0]})"
    
    return {
        "score": score,
        "category": category,
        "label": label,
        "color": color,
        "description": description,
        "thresholds": {
            "positive": positive_threshold,
            "negative": negative_threshold,
            "neutral_range": neutral_range
        }
    }

def get_sentiment_badge_class(score: float, config=None) -> str:
    """
    Get Bootstrap badge class for sentiment score display.
    
    Args:
        score: Sentiment score from -1 to 1
        config: Optional config object, will load if not provided
        
    Returns:
        Bootstrap badge class name
    """
    interpretation = interpret_sentiment_score(score, config)
    return f"bg-{interpretation['color']}"

def get_sentiment_icon(score: float, config=None) -> str:
    """
    Get Bootstrap icon for sentiment score display.
    
    Args:
        score: Sentiment score from -1 to 1
        config: Optional config object, will load if not provided
        
    Returns:
        Bootstrap icon class name
    """
    interpretation = interpret_sentiment_score(score, config)
    
    icon_map = {
        "positive": "bi-arrow-up-circle-fill",
        "negative": "bi-arrow-down-circle-fill",
        "neutral": "bi-dash-circle-fill",
        "weakly_positive": "bi-arrow-up-circle",
        "weakly_negative": "bi-arrow-down-circle"
    }
    
    return icon_map.get(interpretation["category"], "bi-question-circle")

def format_sentiment_display(score: float, config=None) -> str:
    """
    Format sentiment score for display with icon and label.
    
    Args:
        score: Sentiment score from -1 to 1
        config: Optional config object, will load if not provided
        
    Returns:
        Formatted HTML string for display
    """
    interpretation = interpret_sentiment_score(score, config)
    icon = get_sentiment_icon(score, config)
    badge_class = get_sentiment_badge_class(score, config)
    
    return f'<i class="bi {icon}"></i> {interpretation["label"]} ({score:.3f})'

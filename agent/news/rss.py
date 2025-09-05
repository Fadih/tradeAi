from __future__ import annotations

from typing import List
import feedparser

from ..logging_config import get_logger

logger = get_logger(__name__)

def fetch_headlines(feeds: List[str], limit_per_feed: int = 5) -> List[str]:
	logger.info(f"Fetching headlines from {len(feeds)} RSS feeds, limit per feed: {limit_per_feed}")
	
	texts: List[str] = []
	for i, url in enumerate(feeds):
		logger.debug(f"Processing feed {i+1}/{len(feeds)}: {url}")
		
		try:
			logger.debug(f"Parsing RSS feed: {url}")
			feed = feedparser.parse(url)
			
			if not feed.entries:
				logger.warning(f"No entries found in feed: {url}")
				continue
			
			logger.debug(f"Found {len(feed.entries)} entries in feed: {url}")
			
			feed_texts = []
			for j, entry in enumerate(feed.entries[:limit_per_feed]):
				headline = entry.get("title") or ""
				desc = entry.get("summary") or ""
				text = (headline + " - " + desc).strip()
				
				if text:
					feed_texts.append(text)
					logger.debug(f"Entry {j+1}: {headline[:50]}...")
				else:
					logger.debug(f"Entry {j+1}: no valid text content")
			
			logger.info(f"Feed {url}: processed {len(feed_texts)} valid entries")
			texts.extend(feed_texts)
			
		except Exception as e:
			logger.error(f"Failed to process feed {url}: {e}")
			continue
	
	logger.info(f"Total headlines collected: {len(texts)}")
	if texts:
		logger.debug(f"Sample headlines: {texts[:2]}")
	
	return texts



from __future__ import annotations

from typing import List
import feedparser


def fetch_headlines(feeds: List[str], limit_per_feed: int = 5) -> List[str]:
	texts: List[str] = []
	for url in feeds:
		try:
			feed = feedparser.parse(url)
			for entry in feed.entries[:limit_per_feed]:
				headline = entry.get("title") or ""
				desc = entry.get("summary") or ""
				text = (headline + " - " + desc).strip()
				if text:
					texts.append(text)
		except Exception:
			continue
	return texts



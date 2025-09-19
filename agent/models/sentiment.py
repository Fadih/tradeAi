from __future__ import annotations

from typing import List, Optional, Dict, Any
import os
import requests
import aiohttp
import asyncio
import json
import hashlib
from datetime import datetime, timedelta

from ..logging_config import get_logger

try:
	from transformers import pipeline  # optional
	extras_available = True
except Exception:
	extras_available = False

logger = get_logger(__name__)

class SentimentAnalyzer:
	def __init__(self, model_name: str) -> None:
		self.model_name = model_name
		self.hf_token = os.getenv("HF_TOKEN")
		self._pipe = None
		self.cache_ttl = 300  # 5 minutes cache TTL
		
		logger.info(f"Initializing SentimentAnalyzer with model: {model_name}")
		
		if extras_available:
			try:
				logger.debug("Attempting to load local transformers pipeline")
				self._pipe = pipeline("text-classification", model=model_name, truncation=True)
				logger.info("Local transformers pipeline loaded successfully")
			except Exception as e:
				logger.warning(f"Failed to load local pipeline: {e}")
				self._pipe = None
		else:
			logger.info("Transformers not available, will use HF Inference API")
		
		if self.hf_token:
			logger.debug("HF token available for Inference API")
		else:
			logger.warning("No HF token provided, sentiment will fallback to neutral")
	
	def _get_cache_key(self, prefix: str, *args) -> str:
		"""Generate cache key from arguments"""
		key_data = f"{prefix}:{':'.join(str(arg) for arg in args)}"
		return hashlib.md5(key_data.encode()).hexdigest()
	
	async def _get_cached_data(self, cache_key: str) -> Optional[Any]:
		"""Get data from Redis cache"""
		try:
			from agent.cache.redis_client import get_redis_client
			redis_client = await get_redis_client()
			if redis_client:
				cached = await redis_client.get(cache_key)
				if cached:
					return json.loads(cached)
		except Exception as e:
			logger.warning(f"Cache get error: {e}")
		return None
	
	async def _set_cached_data(self, cache_key: str, data: Any) -> None:
		"""Set data in Redis cache"""
		try:
			from agent.cache.redis_client import get_redis_client
			redis_client = await get_redis_client()
			if redis_client:
				await redis_client.setex(cache_key, self.cache_ttl, json.dumps(data, default=str))
		except Exception as e:
			logger.warning(f"Cache set error: {e}")
	
	async def analyze_sentiment_async(self, symbol: str, rss_feeds: List[str], reddit_subreddits: List[str],
									  rss_max_headlines: int, reddit_max_posts: int,
									  rss_hours_back: int = 6, reddit_hours_back: int = 6) -> float:
		"""Async sentiment analysis with caching and parallel data fetching"""
		start_time = datetime.now()
		logger.info(f"🚀 Starting async sentiment analysis for {symbol}")
		
		# Check cache first
		cache_key = self._get_cache_key("sentiment_async", symbol, rss_hours_back, reddit_hours_back)
		cached_result = await self._get_cached_data(cache_key)
		if cached_result:
			duration = (datetime.now() - start_time).total_seconds()
			logger.info(f"📦 Using cached sentiment for {symbol} ({duration:.3f}s)")
			return cached_result
		
		# Import async functions
		from agent.news.rss import fetch_headlines_async
		from agent.news.reddit import fetch_reddit_posts_async
		
		# Fetch RSS and Reddit data in parallel
		fetch_start = datetime.now()
		logger.info(f"🔄 Fetching sentiment data for {symbol} (async)")
		
		rss_task = fetch_headlines_async(
			rss_feeds, rss_max_headlines, symbol, rss_hours_back
		)
		reddit_task = fetch_reddit_posts_async(
			reddit_subreddits, reddit_max_posts, symbol, reddit_hours_back
		)
		
		rss_texts, reddit_posts = await asyncio.gather(rss_task, reddit_task)
		fetch_duration = (datetime.now() - fetch_start).total_seconds()
		logger.info(f"📊 Data fetch completed in {fetch_duration:.3f}s (RSS: {len(rss_texts)}, Reddit: {len(reddit_posts)})")
		
		all_texts = rss_texts + reddit_posts
		
		if not all_texts:
			logger.warning(f"No texts collected for sentiment analysis for {symbol}")
			await self._set_cached_data(cache_key, 0.0)
			return 0.0
		
		# Score sentiment
		score_start = datetime.now()
		sentiment_score = self.score(all_texts)
		score_duration = (datetime.now() - score_start).total_seconds()
		logger.info(f"🧠 Sentiment scoring completed in {score_duration:.3f}s")
		
		# Cache the result
		await self._set_cached_data(cache_key, sentiment_score)
		
		total_duration = (datetime.now() - start_time).total_seconds()
		logger.info(f"✅ Async sentiment analysis completed for {symbol} in {total_duration:.3f}s (score: {sentiment_score:.3f})")
		
		return sentiment_score

	def _score_local(self, texts: List[str]) -> Optional[float]:
		if not self._pipe:
			return None
		
		logger.debug(f"Scoring {len(texts)} texts using local pipeline")
		try:
			results = self._pipe(texts)
			label_to_score = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
			scores = [label_to_score.get(r["label"].lower(), 0.0) for r in results]
			avg_score = float(sum(scores) / len(scores))
			logger.debug(f"Local sentiment scores: {scores}, average: {avg_score:.3f}")
			return avg_score
		except Exception as e:
			logger.error(f"Error in local sentiment scoring: {e}")
			return None

	def _score_hf_inference(self, texts: List[str]) -> Optional[float]:
		if not self.hf_token:
			return None
		
		logger.debug(f"Scoring {len(texts)} texts using HF Inference API")
		api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
		headers = {"Authorization": f"Bearer {self.hf_token}"}
		
		# Batch processing to avoid API limits (max 100 texts per request)
		batch_size = 50
		all_scores = []
		
		for i in range(0, len(texts), batch_size):
			batch = texts[i:i + batch_size]
			logger.debug(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size} with {len(batch)} texts")
			
			try:
				logger.debug(f"Making request to HF Inference API: {self.model_name}")
				resp = requests.post(api_url, headers=headers, json={"inputs": batch}, timeout=30)
				resp.raise_for_status()
				
				payload = resp.json()
				logger.debug(f"Received response from HF API: {len(payload)} items")
				
				# payload can be list-of-list of dicts per input
				label_to_score = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
				batch_scores = []
				
				for j, item in enumerate(payload):
					if isinstance(item, list) and item:
						best = max(item, key=lambda x: x.get("score", 0))
						score = label_to_score.get(best.get("label", "").lower(), 0.0)
						batch_scores.append(score)
						logger.debug(f"Text {i+j}: label='{best.get('label')}', score={score}")
					else:
						batch_scores.append(0.0)
						logger.debug(f"Text {i+j}: no valid response")
				
				all_scores.extend(batch_scores)
				
			except requests.exceptions.Timeout:
				logger.error(f"HF Inference API request timed out for batch {i//batch_size + 1}")
				# Add neutral scores for this batch
				all_scores.extend([0.0] * len(batch))
			except requests.exceptions.RequestException as e:
				logger.error(f"HF Inference API request failed for batch {i//batch_size + 1}: {e}")
				# Add neutral scores for this batch
				all_scores.extend([0.0] * len(batch))
			except Exception as e:
				logger.error(f"Unexpected error in HF Inference scoring for batch {i//batch_size + 1}: {e}")
				# Add neutral scores for this batch
				all_scores.extend([0.0] * len(batch))
		
		if all_scores:
			avg_score = float(sum(all_scores) / len(all_scores))
			logger.info(f"HF Inference sentiment scores: {len(all_scores)} texts processed, average: {avg_score:.3f}")
			return avg_score
		else:
			logger.warning("No valid sentiment scores from HF API")
			return 0.0

	def score(self, texts: List[str]) -> float:
		if not texts:
			logger.warning("No texts provided for sentiment scoring")
			return 0.0
		
		logger.info(f"Scoring sentiment for {len(texts)} texts")
		
		# Prefer local if available; fallback to HF Inference API
		local = self._score_local(texts)
		if local is not None:
			logger.info(f"Using local sentiment score: {local:.3f}")
			return local
		
		remote = self._score_hf_inference(texts)
		if remote is not None:
			logger.info(f"Using HF Inference sentiment score: {remote:.3f}")
			return remote
		
		logger.warning("Both local and remote sentiment scoring failed, using neutral")
		return 0.0



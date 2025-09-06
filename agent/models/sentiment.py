from __future__ import annotations

from typing import List, Optional

import os
import requests

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



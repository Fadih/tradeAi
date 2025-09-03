from __future__ import annotations

from typing import List, Optional

import os
import requests

try:
	from transformers import pipeline  # optional
	extras_available = True
except Exception:
	extras_available = False


class SentimentAnalyzer:
	def __init__(self, model_name: str) -> None:
		self.model_name = model_name
		self.hf_token = os.getenv("HF_TOKEN")
		self._pipe = None
		if extras_available:
			try:
				self._pipe = pipeline("text-classification", model=model_name, truncation=True)
			except Exception:
				self._pipe = None

	def _score_local(self, texts: List[str]) -> Optional[float]:
		if not self._pipe:
			return None
		results = self._pipe(texts)
		label_to_score = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
		scores = [label_to_score.get(r["label"].lower(), 0.0) for r in results]
		return float(sum(scores) / len(scores))

	def _score_hf_inference(self, texts: List[str]) -> Optional[float]:
		if not self.hf_token:
			return None
		api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
		headers = {"Authorization": f"Bearer {self.hf_token}"}
		try:
			resp = requests.post(api_url, headers=headers, json={"inputs": texts})
			resp.raise_for_status()
			payload = resp.json()
			# payload can be list-of-list of dicts per input
			label_to_score = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
			per_input = []
			for item in payload:
				if isinstance(item, list) and item:
					best = max(item, key=lambda x: x.get("score", 0))
					per_input.append(label_to_score.get(best.get("label", "").lower(), 0.0))
				else:
					per_input.append(0.0)
			return float(sum(per_input) / len(per_input)) if per_input else 0.0
		except Exception:
			return None

	def score(self, texts: List[str]) -> float:
		if not texts:
			return 0.0
		# Prefer local if available; fallback to HF Inference API
		local = self._score_local(texts)
		if local is not None:
			return local
		remote = self._score_hf_inference(texts)
		if remote is not None:
			return remote
		return 0.0



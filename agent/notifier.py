from __future__ import annotations

from typing import Iterable
import requests

from .config import NotifierConfig
from .engine import Tip


def send_notifications(config: NotifierConfig, tips: Iterable[Tip]) -> None:
	if config.mode == "slack" and config.slack_webhook_url:
		for tip in tips:
			text = f"[TIP] {tip.symbol} @ {tip.timeframe} | {tip.indicator}={tip.value:.2f} -> {tip.suggestion}"
			if tip.meta:
				text += f" | stop={tip.meta.get('stop'):.2f} tp={tip.meta.get('tp'):.2f}"
			requests.post(config.slack_webhook_url, json={"text": text}, timeout=5)
		return

	if config.mode == "telegram" and config.telegram_token and config.telegram_chat_id:
		base = f"https://api.telegram.org/bot{config.telegram_token}/sendMessage"
		for tip in tips:
			text = f"[TIP] {tip.symbol} @ {tip.timeframe} | {tip.indicator}={tip.value:.2f} -> {tip.suggestion}"
			if tip.meta:
				text += f" | stop={tip.meta.get('stop'):.2f} tp={tip.meta.get('tp'):.2f}"
			requests.post(base, json={"chat_id": config.telegram_chat_id, "text": text}, timeout=5)
		return

	# Console fallback
	for tip in tips:
		line = f"[TIP] {tip.symbol} @ {tip.timeframe} | {tip.indicator}={tip.value:.2f} -> {tip.suggestion}"
		if tip.meta:
			line += f" | stop={tip.meta.get('stop'):.2f} tp={tip.meta.get('tp'):.2f}"
		print(line)



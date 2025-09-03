from __future__ import annotations

from typing import Callable
from apscheduler.schedulers.background import BackgroundScheduler


def start_scheduler(job_fn: Callable[[], None], cron: str = "*/15 * * * *") -> BackgroundScheduler:
	"""Start a background scheduler with a cron expression (minute granularity)."""
	sched = BackgroundScheduler(timezone="UTC")
	# Simple minute-based cron like */15 for 15m
	minute = cron.split()[0]
	sched.add_job(job_fn, trigger="cron", minute=minute)
	sched.start()
	return sched



from __future__ import annotations

from typing import List, Optional
import feedparser
import re
from datetime import datetime, timedelta, timezone
import time

from ..logging_config import get_logger

logger = get_logger(__name__)

def fetch_headlines(feeds: List[str], limit_per_feed: int = 5, symbol: Optional[str] = None, hours_back: int = 6) -> List[str]:
	logger.info(f"Fetching headlines from {len(feeds)} RSS feeds, limit per feed: {limit_per_feed}")
	if symbol:
		logger.info(f"Filtering headlines for symbol: {symbol}")
	logger.info(f"Time filter: Only headlines from last {hours_back} hours")
	
	# Calculate cutoff time for recent headlines (using UTC)
	cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)
	cutoff_timestamp = cutoff_time.timestamp()
	logger.info(f"Cutoff time (UTC): {cutoff_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
	
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
					# Filter by time (only recent headlines)
					entry_time = entry.get("published_parsed")
					if entry_time:
						try:
							# Convert RSS time tuple to UTC timestamp
							entry_timestamp = time.mktime(entry_time)
							# Convert to UTC datetime for comparison
							entry_datetime = datetime.fromtimestamp(entry_timestamp, tz=timezone.utc)
							
							if entry_datetime < cutoff_time:
								logger.debug(f"Entry {j+1}: filtered out (too old, {entry_datetime.strftime('%Y-%m-%d %H:%M:%S UTC')}): {headline[:50]}...")
								continue
							else:
								logger.debug(f"Entry {j+1}: time OK ({entry_datetime.strftime('%Y-%m-%d %H:%M:%S UTC')}): {headline[:50]}...")
						except (ValueError, TypeError) as e:
							logger.debug(f"Entry {j+1}: time parsing error, including anyway: {e}")
							# If we can't parse the time, include the entry to be safe
					else:
						logger.debug(f"Entry {j+1}: no time info, including anyway: {headline[:50]}...")
					
					# Filter by symbol if provided
					if symbol and not _is_relevant_to_symbol(text, symbol):
						logger.debug(f"Entry {j+1}: filtered out (not relevant to {symbol}): {headline[:50]}...")
						continue
					
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


def _is_relevant_to_symbol(text: str, symbol: str) -> bool:
	"""
	Check if a headline/description is relevant to the given symbol.
	
	Args:
		text: The headline and description text
		symbol: The trading symbol (e.g., "BTC/USDT", "ETH/USDT", "AAPL")
	
	Returns:
		True if the text is relevant to the symbol, False otherwise
	"""
	# Convert to lowercase for case-insensitive matching
	text_lower = text.lower()
	symbol_lower = symbol.lower()
	
	# Extract base symbol (remove trading pair info)
	base_symbol = symbol_lower.split('/')[0] if '/' in symbol_lower else symbol_lower
	
	# Define symbol mappings for better matching
	symbol_mappings = {
		'btc': ['bitcoin', 'btc'],
		'eth': ['ethereum', 'eth'],
		'ada': ['cardano', 'ada'],
		'sol': ['solana', 'sol'],
		'matic': ['polygon', 'matic'],
		'avax': ['avalanche', 'avax'],
		'dot': ['polkadot', 'dot'],
		'link': ['chainlink', 'link'],
		'uni': ['uniswap', 'uni'],
		'aave': ['aave'],
		'comp': ['compound', 'comp'],
		'mkr': ['maker', 'mkr'],
		'snx': ['synthetix', 'snx'],
		'crv': ['curve', 'crv'],
		'1inch': ['1inch'],
		'ftm': ['fantom', 'ftm'],
		'algo': ['algorand', 'algo'],
		'atom': ['cosmos', 'atom'],
		'near': ['near'],
		'flow': ['flow'],
		'icp': ['internet computer', 'icp'],
		'fil': ['filecoin', 'fil'],
		'xtz': ['tezos', 'xtz'],
		'egld': ['multiversx', 'egld'],
		'hbar': ['hedera', 'hbar'],
		'vet': ['vechain', 'vet'],
		'trx': ['tron', 'trx'],
		'ltc': ['litecoin', 'ltc'],
		'bch': ['bitcoin cash', 'bch'],
		'xrp': ['ripple', 'xrp'],
		'doge': ['dogecoin', 'doge'],
		'shib': ['shiba inu', 'shib'],
		'matic': ['polygon', 'matic'],
		'bnb': ['binance coin', 'bnb'],
		'cake': ['pancakeswap', 'cake'],
		'sushi': ['sushiswap', 'sushi'],
		'grt': ['the graph', 'grt'],
		'bat': ['basic attention token', 'bat'],
		'zec': ['zcash', 'zec'],
		'dash': ['dash'],
		'monero': ['monero', 'xmr'],
		'xmr': ['monero', 'xmr'],
		'etc': ['ethereum classic', 'etc'],
		'neo': ['neo'],
		'qtum': ['qtum'],
		'waves': ['waves'],
		'omg': ['omg network', 'omg'],
		'zrx': ['0x protocol', 'zrx'],
		'knc': ['kyber network', 'knc'],
		'ren': ['ren protocol', 'ren'],
		'storj': ['storj'],
		'golem': ['golem', 'glm'],
		'glm': ['golem', 'glm'],
		'mana': ['decentraland', 'mana'],
		'sand': ['sandbox', 'sand'],
		'axs': ['axie infinity', 'axs'],
		'enj': ['enjin', 'enj'],
		'chz': ['chiliz', 'chz'],
		'flow': ['flow'],
		'theta': ['theta'],
		'ftm': ['fantom', 'ftm'],
		'one': ['harmony', 'one'],
		'celo': ['celo'],
		'klay': ['klaytn', 'klay'],
		'icx': ['icon', 'icx'],
		'zil': ['zilliqa', 'zil'],
		'ont': ['ontology', 'ont'],
		'qtum': ['qtum'],
		'ardr': ['ardor', 'ardr'],
		'str': ['stellar', 'xlm'],
		'xlm': ['stellar', 'xlm'],
		'nano': ['nano'],
		'iota': ['iota'],
		'iotx': ['iotex', 'iotx'],
		'hbar': ['hedera', 'hbar'],
		'egld': ['multiversx', 'egld'],
		'elrond': ['multiversx', 'egld'],
		'kava': ['kava'],
		'band': ['band protocol', 'band'],
		'rsr': ['reserve rights', 'rsr'],
		'uma': ['uma'],
		'bal': ['balancer', 'bal'],
		'knc': ['kyber network', 'knc'],
		'ren': ['ren protocol', 'ren'],
		'storj': ['storj'],
		'golem': ['golem', 'glm'],
		'glm': ['golem', 'glm'],
		'mana': ['decentraland', 'mana'],
		'sand': ['sandbox', 'sand'],
		'axs': ['axie infinity', 'axs'],
		'enj': ['enjin', 'enj'],
		'chz': ['chiliz', 'chz'],
		'flow': ['flow'],
		'theta': ['theta'],
		'ftm': ['fantom', 'ftm'],
		'one': ['harmony', 'one'],
		'celo': ['celo'],
		'klay': ['klaytn', 'klay'],
		'icx': ['icon', 'icx'],
		'zil': ['zilliqa', 'zil'],
		'ont': ['ontology', 'ont'],
		'qtum': ['qtum'],
		'ardr': ['ardor', 'ardr'],
		'str': ['stellar', 'xlm'],
		'xlm': ['stellar', 'xlm'],
		'nano': ['nano'],
		'iota': ['iota'],
		'iotx': ['iotex', 'iotx'],
		'hbar': ['hedera', 'hbar'],
		'egld': ['multiversx', 'egld'],
		'elrond': ['multiversx', 'egld'],
		'kava': ['kava'],
		'band': ['band protocol', 'band'],
		'rsr': ['reserve rights', 'rsr'],
		'uma': ['uma'],
		'bal': ['balancer', 'bal'],
		# Stock symbols (common ones)
		'aapl': ['apple', 'aapl'],
		'msft': ['microsoft', 'msft'],
		'googl': ['google', 'alphabet', 'googl'],
		'amzn': ['amazon', 'amzn'],
		'tsla': ['tesla', 'tsla'],
		'meta': ['facebook', 'meta'],
		'nvda': ['nvidia', 'nvda'],
		'brk.a': ['berkshire hathaway', 'berkshire', 'brk'],
		'brk.b': ['berkshire hathaway', 'berkshire', 'brk'],
		'v': ['visa', 'v'],
		'jpm': ['jpmorgan', 'jpm'],
		'wmt': ['walmart', 'wmt'],
		'pg': ['procter & gamble', 'procter', 'gamble', 'pg'],
		'jnj': ['johnson & johnson', 'johnson', 'jnj'],
		'hd': ['home depot', 'hd'],
		'ma': ['mastercard', 'ma'],
		'dis': ['disney', 'dis'],
		'pypl': ['paypal', 'pypl'],
		'adbe': ['adobe', 'adbe'],
		'cmcsa': ['comcast', 'cmcsa'],
		'vz': ['verizon', 'vz'],
		't': ['at&t', 'att', 't'],
		'crm': ['salesforce', 'crm'],
		'abbv': ['abbvie', 'abbv'],
		'ko': ['coca cola', 'coca-cola', 'ko'],
		'pep': ['pepsico', 'pep'],
		'mrk': ['merck', 'mrk'],
		'pfe': ['pfizer', 'pfe'],
		'wfc': ['wells fargo', 'wfc'],
		'c': ['citigroup', 'c'],
		'gs': ['goldman sachs', 'gs'],
		'axp': ['american express', 'amex', 'axp'],
		'ibm': ['ibm'],
		'ge': ['general electric', 'ge'],
		'cat': ['caterpillar', 'cat'],
		'ba': ['boeing', 'ba'],
		'mcd': ['mcdonalds', 'mcd'],
		'nke': ['nike', 'nke'],
		'sbux': ['starbucks', 'sbux'],
		'cost': ['costco', 'cost'],
		'tgt': ['target', 'tgt'],
		'lmt': ['lockheed martin', 'lmt'],
		'rtx': ['raytheon', 'rtx'],
		'de': ['deere', 'de'],
		'hon': ['honeywell', 'hon'],
		'ups': ['ups'],
		'fedex': ['fedex'],
		'fdx': ['fedex', 'fdx'],
		'cop': ['conocophillips', 'cop'],
		'xom': ['exxon mobil', 'exxon', 'xom'],
		'cvx': ['chevron', 'cvx'],
		'slb': ['schlumberger', 'slb'],
		'eog': ['eog resources', 'eog'],
		'pfe': ['pfizer', 'pfe'],
		'mrk': ['merck', 'mrk'],
		'jnj': ['johnson & johnson', 'johnson', 'jnj'],
		'abbv': ['abbvie', 'abbv'],
		'gild': ['gilead', 'gild'],
		'amgn': ['amgen', 'amgn'],
		'bmy': ['bristol myers', 'bristol-myers', 'bmy'],
		'celg': ['celgene', 'celg'],
		'regn': ['regeneron', 'regn'],
		'vrtx': ['vertex', 'vrtx'],
		'biib': ['biogen', 'biib'],
		'ilmn': ['illumina', 'ilmn'],
		'mrna': ['moderna', 'mrna'],
		'bntx': ['biontech', 'bntx'],
		'pfe': ['pfizer', 'pfe'],
		'jnj': ['johnson & johnson', 'johnson', 'jnj'],
		'mrk': ['merck', 'mrk'],
		'abbv': ['abbvie', 'abbv'],
		'gild': ['gilead', 'gild'],
		'amgn': ['amgen', 'amgn'],
		'bmy': ['bristol myers', 'bristol-myers', 'bmy'],
		'celg': ['celgene', 'celg'],
		'regn': ['regeneron', 'regn'],
		'vrtx': ['vertex', 'vrtx'],
		'biib': ['biogen', 'biib'],
		'ilmn': ['illumina', 'ilmn'],
		'mrna': ['moderna', 'mrna'],
		'bntx': ['biontech', 'bntx'],
	}
	
	# Get search terms for the symbol
	search_terms = symbol_mappings.get(base_symbol, [base_symbol])
	
	# Check if any search term appears in the text
	for term in search_terms:
		if term in text_lower:
			return True
	
	# Also check for exact symbol match (case insensitive)
	if base_symbol in text_lower:
		return True
	
	# For crypto symbols, also check common variations
	if base_symbol in ['btc', 'bitcoin']:
		return any(term in text_lower for term in ['bitcoin', 'btc', 'crypto', 'cryptocurrency'])
	elif base_symbol in ['eth', 'ethereum']:
		return any(term in text_lower for term in ['ethereum', 'eth', 'crypto', 'cryptocurrency'])
	
	return False



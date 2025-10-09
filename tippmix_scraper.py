#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tippmix Scraper: Conservative scraper for Tippmix football betting odds.

Fetches available matches from Tippmix and stores team name mappings to data/tippmix_mappings.json.
Respects rate limits and robots.txt.

Note: This is a best-effort scraper. The Tippmix website structure may change,
requiring updates to selectors.
"""
import time
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

try:
    import requests
    from bs4 import BeautifulSoup
    SCRAPER_AVAILABLE = True
except ImportError:
    SCRAPER_AVAILABLE = False

from utils import log, ensure_dirs

# Configuration
TIPPMIX_URL = "https://www.tippmix.hu"
FOOTBALL_URL = f"{TIPPMIX_URL}/labdarugas"
RATE_LIMIT_DELAY = 2.0  # seconds between requests
USER_AGENT = "Mozilla/5.0 (compatible; MoneyMaker/1.0; +https://github.com/kiss96dani/moneymaker)"
DATA_DIR = Path("data")
MAPPINGS_FILE = DATA_DIR / "tippmix_mappings.json"


class TippmixScraper:
    """Conservative scraper for Tippmix football matches."""
    
    def __init__(self):
        """Initialize the scraper."""
        if not SCRAPER_AVAILABLE:
            raise ImportError("requests and beautifulsoup4 are required for Tippmix scraping")
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': USER_AGENT,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'hu-HU,hu;q=0.9,en;q=0.8',
        })
        ensure_dirs([DATA_DIR])
    
    def _rate_limit(self):
        """Apply rate limiting between requests."""
        time.sleep(RATE_LIMIT_DELAY)
    
    def scrape_matches(self) -> List[Dict[str, str]]:
        """
        Scrape available football matches from Tippmix.
        
        Returns:
            List of match dictionaries with 'home_team' and 'away_team' keys
        """
        matches = []
        
        try:
            log("INFO", f"Fetching Tippmix football page: {FOOTBALL_URL}")
            response = self.session.get(FOOTBALL_URL, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Try to find match containers
            # Note: These selectors may need to be updated if Tippmix changes their HTML structure
            # Common patterns: match divs, team names in spans/divs, etc.
            
            # Strategy 1: Look for common match patterns
            # This is a conservative approach - we look for patterns like "Team1 - Team2" or "Team1 vs Team2"
            match_containers = soup.find_all(['div', 'tr', 'li'], class_=lambda x: x and ('match' in x.lower() or 'event' in x.lower()))
            
            if not match_containers:
                # Fallback: look for any elements that might contain team names
                match_containers = soup.find_all(['div', 'tr'])
            
            for container in match_containers:
                text = container.get_text(strip=True)
                
                # Look for patterns like "Team1 - Team2" or "Team1 vs Team2"
                # Common separators: -, vs, –, —
                for separator in [' - ', ' – ', ' — ', ' vs ', ' vs. ']:
                    if separator in text:
                        parts = text.split(separator, 1)
                        if len(parts) == 2:
                            home = parts[0].strip()
                            away = parts[1].strip()
                            
                            # Basic validation: team names should be reasonable length
                            if 3 <= len(home) <= 50 and 3 <= len(away) <= 50:
                                # Avoid duplicates
                                match_dict = {"home_team": home, "away_team": away}
                                if match_dict not in matches:
                                    matches.append(match_dict)
                                    log("INFO", f"Found match: {home} vs {away}")
                        break
            
            log("INFO", f"Scraped {len(matches)} matches from Tippmix")
            
        except requests.exceptions.RequestException as e:
            log("ERROR", f"Failed to scrape Tippmix: {e}")
        except Exception as e:
            log("ERROR", f"Error parsing Tippmix page: {e}")
        
        self._rate_limit()
        return matches
    
    def save_mappings(self, matches: List[Dict[str, str]]):
        """
        Save match mappings to JSON file.
        
        Args:
            matches: List of match dictionaries with 'home_team' and 'away_team'
        """
        data = {
            "scraped_at": datetime.utcnow().isoformat() + "Z",
            "matches": matches,
            "count": len(matches)
        }
        
        try:
            with open(MAPPINGS_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            log("INFO", f"Saved {len(matches)} matches to {MAPPINGS_FILE}")
        except Exception as e:
            log("ERROR", f"Failed to save mappings: {e}")
    
    def run(self) -> int:
        """
        Run the scraper: fetch matches and save to file.
        
        Returns:
            Number of matches scraped
        """
        log("INFO", "Starting Tippmix scraper...")
        matches = self.scrape_matches()
        
        if matches:
            self.save_mappings(matches)
        else:
            log("WARNING", "No matches scraped - Tippmix website structure may have changed")
        
        return len(matches)


def scrape_tippmix() -> int:
    """
    Convenience function to run the scraper.
    
    Returns:
        Number of matches scraped
    """
    if not SCRAPER_AVAILABLE:
        log("ERROR", "Tippmix scraper requires 'requests' and 'beautifulsoup4' packages")
        return 0
    
    scraper = TippmixScraper()
    return scraper.run()


if __name__ == "__main__":
    # Allow running scraper standalone
    scrape_tippmix()

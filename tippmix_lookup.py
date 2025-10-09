#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tippmix Lookup: Fuzzy matching to check if a match is available on Tippmix.

Loads scraped data from data/tippmix_mappings.json and uses fuzzy matching
(rapidfuzz) to determine if a given home/away team pair is on Tippmix.
"""
import json
from pathlib import Path
from typing import Optional, Tuple

try:
    from rapidfuzz import process, fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

from utils import log

# Configuration
DATA_DIR = Path("data")
MAPPINGS_FILE = DATA_DIR / "tippmix_mappings.json"
FUZZY_THRESHOLD = 85  # Minimum similarity score (0-100) for a match


class TippmixLookup:
    """Fuzzy matching lookup for Tippmix availability."""
    
    def __init__(self, auto_scrape: bool = False):
        """
        Initialize the lookup.
        
        Args:
            auto_scrape: If True and mappings not found, automatically run scraper
        """
        self.matches = []
        self.team_names = []
        
        if not RAPIDFUZZ_AVAILABLE:
            log("WARNING", "rapidfuzz not available - fuzzy matching disabled")
        
        self.load_mappings(auto_scrape=auto_scrape)
    
    def load_mappings(self, auto_scrape: bool = False):
        """
        Load Tippmix mappings from JSON file.
        
        Args:
            auto_scrape: If True and file not found, run scraper automatically
        """
        if not MAPPINGS_FILE.exists():
            log("WARNING", f"Tippmix mappings file not found: {MAPPINGS_FILE}")
            
            if auto_scrape:
                log("INFO", "Attempting to run Tippmix scraper automatically...")
                try:
                    from tippmix_scraper import scrape_tippmix
                    count = scrape_tippmix()
                    if count == 0:
                        log("WARNING", "Auto-scrape returned no matches")
                        return
                except Exception as e:
                    log("ERROR", f"Auto-scrape failed: {e}")
                    return
            else:
                log("INFO", "Run 'python betting.py --tippmix-scrape' to fetch Tippmix data")
                return
        
        try:
            with open(MAPPINGS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.matches = data.get('matches', [])
            
            # Build a list of all team names for fuzzy matching
            self.team_names = []
            for match in self.matches:
                home = match.get('home_team', '')
                away = match.get('away_team', '')
                if home:
                    self.team_names.append(home)
                if away:
                    self.team_names.append(away)
            
            log("INFO", f"Loaded {len(self.matches)} matches from Tippmix mappings")
            
        except Exception as e:
            log("ERROR", f"Failed to load Tippmix mappings: {e}")
    
    def fuzzy_match_team(self, query: str) -> Optional[Tuple[str, float]]:
        """
        Find the best fuzzy match for a team name.
        
        Args:
            query: Team name to search for
            
        Returns:
            Tuple of (matched_name, score) or None if no good match
        """
        if not RAPIDFUZZ_AVAILABLE or not self.team_names:
            return None
        
        # Use rapidfuzz to find best match
        result = process.extractOne(
            query,
            self.team_names,
            scorer=fuzz.ratio,
            score_cutoff=FUZZY_THRESHOLD
        )
        
        if result:
            matched_name, score, _ = result
            return (matched_name, score)
        
        return None
    
    def is_on_tippmix(self, home_team: str, away_team: str) -> bool:
        """
        Check if a match (home vs away) is available on Tippmix.
        
        Uses fuzzy matching to handle slight variations in team names.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            
        Returns:
            True if match is found on Tippmix, False otherwise
        """
        if not self.matches:
            # No data loaded, can't determine
            return False
        
        if not RAPIDFUZZ_AVAILABLE:
            # Fallback to exact matching
            for match in self.matches:
                if match.get('home_team') == home_team and match.get('away_team') == away_team:
                    return True
            return False
        
        # Fuzzy match home team
        home_match = self.fuzzy_match_team(home_team)
        if not home_match:
            return False
        
        home_matched_name, home_score = home_match
        
        # Fuzzy match away team
        away_match = self.fuzzy_match_team(away_team)
        if not away_match:
            return False
        
        away_matched_name, away_score = away_match
        
        # Check if these matched teams form a match in our data
        for match in self.matches:
            if match.get('home_team') == home_matched_name and match.get('away_team') == away_matched_name:
                log("INFO", f"Match found on Tippmix: {home_team} ({home_score:.0f}% → {home_matched_name}) vs {away_team} ({away_score:.0f}% → {away_matched_name})")
                return True
        
        return False


# Global instance for easy access
_lookup_instance: Optional[TippmixLookup] = None


def get_lookup(auto_scrape: bool = False) -> TippmixLookup:
    """
    Get or create the global TippmixLookup instance.
    
    Args:
        auto_scrape: If True and mappings not found, automatically run scraper
        
    Returns:
        TippmixLookup instance
    """
    global _lookup_instance
    if _lookup_instance is None:
        _lookup_instance = TippmixLookup(auto_scrape=auto_scrape)
    return _lookup_instance


def is_on_tippmix(home_team: str, away_team: str, auto_scrape: bool = False) -> bool:
    """
    Convenience function to check if a match is on Tippmix.
    
    Args:
        home_team: Home team name
        away_team: Away team name
        auto_scrape: If True and mappings not found, automatically run scraper
        
    Returns:
        True if match is found on Tippmix, False otherwise
    """
    lookup = get_lookup(auto_scrape=auto_scrape)
    return lookup.is_on_tippmix(home_team, away_team)

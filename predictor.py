from typing import Dict, Tuple, List, Optional
import math
import os
from collections import defaultdict

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

HOME_ADV = float(os.getenv("HOME_ADV", "0.20"))
BANKROLL_DAILY = float(os.getenv("BANKROLL_DAILY", "1000"))

def _poisson_pmf(l: float, k: int) -> float:
    # Poisson probability mass function
    try:
        return (l ** k) * math.exp(-l) / math.factorial(k)
    except OverflowError:
        return 0.0

def _calculate_form_score(form_string: str) -> float:
    """
    Calculate weighted form score from result string.
    W=1, D=0.5, L=0. Recent matches weighted more.
    """
    if not form_string:
        return 0.5
    
    weights = [0.6, 0.7, 0.8, 0.9, 1.0]  # More recent = higher weight
    score = 0.0
    total_weight = 0.0
    
    # Reverse to process oldest to newest
    results = list(reversed(form_string[-5:]))
    for i, res in enumerate(results):
        w = weights[min(i, len(weights) - 1)]
        if res == 'W':
            score += 1.0 * w
        elif res == 'D':
            score += 0.5 * w
        total_weight += w
    
    return score / total_weight if total_weight > 0 else 0.5

def monte_carlo_simulation(lambda_h: float, lambda_a: float, iters: int = 10000) -> Dict[str, any]:
    """
    Perform Monte Carlo simulation using Poisson sampling for match outcomes.
    
    Args:
        lambda_h: Home team expected goals (lambda parameter)
        lambda_a: Away team expected goals (lambda parameter)
        iters: Number of simulations to run (default: 10000)
    
    Returns:
        Dictionary containing:
        - Empirical probabilities for 1X2, BTTS, Over/Under 2.5
        - Total goals distribution quantiles (5, 25, 50, 75, 95 percentiles)
    """
    if not NUMPY_AVAILABLE:
        raise ImportError("NumPy is required for Monte Carlo simulation")
    
    # Sample goals from Poisson distributions
    home_goals = np.random.poisson(lambda_h, iters)
    away_goals = np.random.poisson(lambda_a, iters)
    
    # Calculate outcomes
    home_wins = np.sum(home_goals > away_goals)
    draws = np.sum(home_goals == away_goals)
    away_wins = np.sum(home_goals < away_goals)
    
    # BTTS (Both Teams To Score)
    btts_yes = np.sum((home_goals > 0) & (away_goals > 0))
    
    # Over/Under 2.5
    total_goals = home_goals + away_goals
    over25 = np.sum(total_goals > 2.5)
    
    # Calculate probabilities
    p_home = home_wins / iters
    p_draw = draws / iters
    p_away = away_wins / iters
    p_btts_yes = btts_yes / iters
    p_over25 = over25 / iters
    
    # Calculate total goals quantiles
    quantiles = np.percentile(total_goals, [5, 25, 50, 75, 95])
    
    return {
        "home": round(float(p_home), 4),
        "draw": round(float(p_draw), 4),
        "away": round(float(p_away), 4),
        "btts_yes": round(float(p_btts_yes), 4),
        "btts_no": round(float(1.0 - p_btts_yes), 4),
        "over25": round(float(p_over25), 4),
        "under25": round(float(1.0 - p_over25), 4),
        "total_goals_quantiles": {
            "p5": round(float(quantiles[0]), 2),
            "p25": round(float(quantiles[1]), 2),
            "p50": round(float(quantiles[2]), 2),
            "p75": round(float(quantiles[3]), 2),
            "p95": round(float(quantiles[4]), 2),
        }
    }

class Predictor:
    def __init__(self, use_ml: bool = False):
        """
        Initialize Predictor with optional ML mode.
        
        Args:
            use_ml: If True, attempt to use ML models for predictions
        """
        self.use_ml = use_ml
        self.ml_model = None
        
        if use_ml:
            try:
                from models.model_wrapper import ModelWrapper
                self.ml_model = ModelWrapper()
                if self.ml_model.load_models():
                    from utils import log
                    log("INFO", "Predictor initialized with ML models")
                else:
                    self.ml_model = None
                    from utils import log
                    log("WARNING", "ML models not available, falling back to Poisson")
            except Exception as e:
                from utils import log
                log("WARNING", f"Failed to initialize ML models: {e}, using Poisson fallback")
                self.ml_model = None

    def compute_lambdas(self, home_recent: Dict, away_recent: Dict) -> Tuple[float, float]:
        # A simple lambda calculation using goals_per_match and home advantage
        gh = home_recent.get("goals_per_match", 0.0)
        ga = away_recent.get("goals_per_match", 0.0)
        lambda_h = max(0.01, gh * (1.0 + HOME_ADV))
        lambda_a = max(0.01, ga)
        return round(lambda_h, 3), round(lambda_a, 3)

    def match_probabilities(
        self, 
        lambda_h: float, 
        lambda_a: float, 
        max_goals: int = 6,
        home_form: Optional[Dict] = None,
        away_form: Optional[Dict] = None,
        use_mc: bool = False,
        mc_iters: int = 10000
    ) -> Tuple[Dict[str, float], List[List[float]]]:
        """
        Compute full score probability matrix up to max_goals each,
        then aggregate to 1X2, BTTS and Over/Under 2.5.
        
        If ML mode is enabled and models are available, uses ML predictions.
        If use_mc is True and NumPy is available, uses Monte Carlo simulation.
        Otherwise falls back to Poisson distribution.
        
        Args:
            lambda_h: Home team lambda (expected goals)
            lambda_a: Away team lambda (expected goals)
            max_goals: Maximum goals to consider in Poisson
            home_form: Home team form data (for ML mode)
            away_form: Away team form data (for ML mode)
            use_mc: Whether to use Monte Carlo simulation (default: False)
            mc_iters: Number of Monte Carlo iterations (default: 10000)
        """
        # Try ML prediction if enabled
        if self.use_ml and self.ml_model and self.ml_model.is_available():
            if home_form and away_form:
                features = {
                    'home_goals_per_match': home_form.get('goals_per_match', lambda_h),
                    'home_goals_against_per_match': home_form.get('goals_against_per_match', 1.0),
                    'away_goals_per_match': away_form.get('goals_per_match', lambda_a),
                    'away_goals_against_per_match': away_form.get('goals_against_per_match', 1.0),
                    'form_score_home': _calculate_form_score(home_form.get('form_string', '')),
                    'form_score_away': _calculate_form_score(away_form.get('form_string', '')),
                    'home_adv': HOME_ADV
                }
                
                ml_probs = self.ml_model.predict_proba(features)
                if ml_probs:
                    # Still compute matrix for compatibility
                    matrix = [[0.0]*(max_goals+1) for _ in range(max_goals+1)]
                    return ml_probs, matrix
        
        # Try Monte Carlo simulation if requested and available
        if use_mc and NUMPY_AVAILABLE:
            try:
                mc_probs = monte_carlo_simulation(lambda_h, lambda_a, mc_iters)
                # Still compute matrix for compatibility
                matrix = [[0.0]*(max_goals+1) for _ in range(max_goals+1)]
                return mc_probs, matrix
            except Exception as e:
                from utils import log
                log("WARNING", f"Monte Carlo simulation failed: {e}, falling back to Poisson")
        
        # Fallback to Poisson distribution
        # build distributions
        hd = [_poisson_pmf(lambda_h, k) for k in range(0, max_goals+1)]
        ad = [_poisson_pmf(lambda_a, k) for k in range(0, max_goals+1)]

        # matrix
        matrix = [[0.0]*(max_goals+1) for _ in range(max_goals+1)]
        p_home = p_draw = p_away = 0.0
        p_btts_yes = 0.0
        p_over25 = 0.0

        for i in range(0, max_goals+1):
            for j in range(0, max_goals+1):
                prob = hd[i] * ad[j]
                matrix[i][j] = prob
                if i > j:
                    p_home += prob
                elif i == j:
                    p_draw += prob
                else:
                    p_away += prob
                if i > 0 and j > 0:
                    p_btts_yes += prob
                if (i + j) > 2:
                    p_over25 += prob

        # account for tail (goals > max_goals) approx by 1 - sum
        total_mass = sum(sum(row) for row in matrix)
        tail = max(0.0, 1.0 - total_mass)
        # approximate tail to draw proportionally to total (simple)
        p_home += tail * 0.45
        p_draw += tail * 0.10
        p_away += tail * 0.45
        p_over25 = min(1.0, p_over25 + tail * 0.5)
        p_btts_yes = min(1.0, p_btts_yes + tail * 0.5)

        model_probs = {
            "home": round(p_home, 4),
            "draw": round(p_draw, 4),
            "away": round(p_away, 4),
            "btts_yes": round(p_btts_yes, 4),
            "btts_no": round(1.0 - p_btts_yes, 4),
            "over25": round(p_over25, 4),
            "under25": round(1.0 - p_over25, 4),
        }
        return model_probs, matrix

    def parse_market_odds(self, odds_response: List[dict]) -> Dict:
        """
        Parse bookmaker odds for 1X2, BTTS and O/U 2.5 markets.
        Returns dict with market_odds (decimal odds) and market_probs (implied probabilities).
        """
        market_odds = {}
        
        # odds_response format: list of objects with bookmakers[].bets[].values[]
        for venue in odds_response:
            # accommodate different shapes
            bookmakers = []
            if isinstance(venue, dict) and "bookmakers" in venue:
                bookmakers = venue.get("bookmakers", [])
            elif isinstance(venue, dict) and "bookmaker" in venue:
                bookmakers = venue.get("bookmaker", [])
            else:
                bookmakers = [venue]

            for bm in bookmakers:
                for bet in bm.get("bets", []):
                    label = bet.get("name", "") or bet.get("label", "")
                    
                    # Parse 1X2 odds
                    if "1X2" in label.upper() or "MATCH WINNER" in label.upper() or "MATCH ODDS" in label.upper():
                        for v in bet.get("values", []):
                            value_key = (v.get("value", "") or "").strip()
                            odd = v.get("odd")
                            
                            if not odd:
                                continue
                            
                            try:
                                odd_f = float(odd)
                                if value_key == "Home" or value_key == "1":
                                    market_odds.setdefault("home_odds", odd_f)
                                elif value_key == "Draw" or value_key == "X":
                                    market_odds.setdefault("draw_odds", odd_f)
                                elif value_key == "Away" or value_key == "2":
                                    market_odds.setdefault("away_odds", odd_f)
                            except (ValueError, TypeError):
                                continue
                    
                    # Parse BTTS odds
                    if "BTTS" in label.upper() or "BOTH TEAMS" in label.upper() or "BOTH TEAMS TO SCORE" in label.upper():
                        for v in bet.get("values", []):
                            value_key = (v.get("value", "") or "").lower()
                            odd = v.get("odd")
                            
                            if not odd:
                                continue
                            
                            try:
                                odd_f = float(odd)
                                if "yes" in value_key or value_key == "y":
                                    market_odds.setdefault("btts_yes_odds", odd_f)
                                elif "no" in value_key or value_key == "n":
                                    market_odds.setdefault("btts_no_odds", odd_f)
                            except (ValueError, TypeError):
                                continue
                    
                    # Parse Over/Under 2.5 odds
                    if "OVER/UNDER" in label.upper() or "GOALS OVER/UNDER" in label.upper():
                        for v in bet.get("values", []):
                            value_key = (v.get("value", "") or "").lower()
                            odd = v.get("odd")
                            
                            if "2.5" in value_key and odd:
                                try:
                                    odd_f = float(odd)
                                    if "over" in value_key:
                                        market_odds.setdefault("over25_odds", odd_f)
                                    elif "under" in value_key:
                                        market_odds.setdefault("under25_odds", odd_f)
                                except (ValueError, TypeError):
                                    continue
        
        # Convert odds to implied probabilities
        market_probs = {}
        
        # 1X2 probabilities
        if all(k in market_odds for k in ["home_odds", "draw_odds", "away_odds"]):
            market_probs["home"] = round(1.0 / market_odds["home_odds"], 4)
            market_probs["draw"] = round(1.0 / market_odds["draw_odds"], 4)
            market_probs["away"] = round(1.0 / market_odds["away_odds"], 4)
        
        # BTTS probabilities
        if "btts_yes_odds" in market_odds:
            market_probs["btts_yes"] = round(1.0 / market_odds["btts_yes_odds"], 4)
        if "btts_no_odds" in market_odds:
            market_probs["btts_no"] = round(1.0 / market_odds["btts_no_odds"], 4)
        
        # Over/Under 2.5 probabilities
        if "over25_odds" in market_odds:
            market_probs["over25"] = round(1.0 / market_odds["over25_odds"], 4)
        if "under25_odds" in market_odds:
            market_probs["under25"] = round(1.0 / market_odds["under25_odds"], 4)
        
        # Return both odds and probabilities
        return {
            "market_odds": market_odds,
            "market_probs": market_probs
        }

    def compute_edges_and_kelly(self, model_probs: Dict[str, float], market_data: Dict) -> Dict:
        """
        For each market available, compute edge and Kelly stake %.
        edge = model_prob * odds - 1
        kelly = edge / (odds - 1) if odds>1 and edge>0 else 0
        
        Args:
            model_probs: Model predicted probabilities
            market_data: Dict with 'market_odds' and 'market_probs' from parse_market_odds
        """
        res = {}
        
        # Handle both old and new format
        market_odds = market_data.get("market_odds", {})
        market_probs = market_data.get("market_probs", market_data)  # Fallback for old format
        
        # Process 1X2 markets
        for outcome in ["home", "draw", "away"]:
            odds_key = f"{outcome}_odds"
            if outcome in model_probs and odds_key in market_odds:
                odds = market_odds[odds_key]
                model_p = model_probs[outcome]
                
                if odds > 1.0:
                    edge = model_p * odds - 1.0
                    kelly = (edge / (odds - 1.0)) if (odds - 1.0) > 0 and edge > 0 else 0.0
                    stake = round(max(0.0, min(1.0, kelly)) * BANKROLL_DAILY, 2)
                    
                    res[outcome] = {
                        "market_prob": round(1.0 / odds, 4),
                        "model_prob": round(model_p, 4),
                        "odds": round(odds, 3),
                        "edge": round(edge, 4),
                        "kelly_frac": round(kelly, 4),
                        "stake_recom": stake,
                    }
        
        # Process BTTS markets
        for outcome in ["btts_yes", "btts_no"]:
            odds_key = outcome + "_odds"
            if outcome in model_probs and odds_key in market_odds:
                odds = market_odds[odds_key]
                model_p = model_probs[outcome]
                
                if odds > 1.0:
                    edge = model_p * odds - 1.0
                    kelly = (edge / (odds - 1.0)) if (odds - 1.0) > 0 and edge > 0 else 0.0
                    stake = round(max(0.0, min(1.0, kelly)) * BANKROLL_DAILY, 2)
                    
                    res[outcome] = {
                        "market_prob": round(1.0 / odds, 4),
                        "model_prob": round(model_p, 4),
                        "odds": round(odds, 3),
                        "edge": round(edge, 4),
                        "kelly_frac": round(kelly, 4),
                        "stake_recom": stake,
                    }
        
        # Process Over/Under 2.5 markets
        for outcome in ["over25", "under25"]:
            odds_key = outcome + "_odds"
            if outcome in model_probs and odds_key in market_odds:
                odds = market_odds[odds_key]
                model_p = model_probs[outcome]
                
                if odds > 1.0:
                    edge = model_p * odds - 1.0
                    kelly = (edge / (odds - 1.0)) if (odds - 1.0) > 0 and edge > 0 else 0.0
                    stake = round(max(0.0, min(1.0, kelly)) * BANKROLL_DAILY, 2)
                    
                    res[outcome] = {
                        "market_prob": round(1.0 / odds, 4),
                        "model_prob": round(model_p, 4),
                        "odds": round(odds, 3),
                        "edge": round(edge, 4),
                        "kelly_frac": round(kelly, 4),
                        "stake_recom": stake,
                    }
        
        return res
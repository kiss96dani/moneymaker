from typing import Dict, Tuple, List
import math
import os
from collections import defaultdict

HOME_ADV = float(os.getenv("HOME_ADV", "0.20"))
BANKROLL_DAILY = float(os.getenv("BANKROLL_DAILY", "1000"))

def _poisson_pmf(l: float, k: int) -> float:
    # Poisson probability mass function
    try:
        return (l ** k) * math.exp(-l) / math.factorial(k)
    except OverflowError:
        return 0.0

class Predictor:
    def __init__(self):
        pass

    def compute_lambdas(self, home_recent: Dict, away_recent: Dict) -> Tuple[float, float]:
        # A simple lambda calculation using goals_per_match and home advantage
        gh = home_recent.get("goals_per_match", 0.0)
        ga = away_recent.get("goals_per_match", 0.0)
        lambda_h = max(0.01, gh * (1.0 + HOME_ADV))
        lambda_a = max(0.01, ga)
        return round(lambda_h, 3), round(lambda_a, 3)

    def match_probabilities(self, lambda_h: float, lambda_a: float, max_goals: int = 6) -> Tuple[Dict[str, float], List[List[float]]]:
        """
        Compute full score probability matrix up to max_goals each,
        then aggregate to 1X2, BTTS and Over/Under 2.5.
        """
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
        Simplified parsing: try to find common bookmakers and odds for 1X2, BTTS and O/U 2.5
        Returns a dict with market probabilities (converted from decimal odds).
        """
        markets = {}
        # odds_response format: list of bookmakers each with bets -> contains bookmakers[].bets[].values[]
        for book in odds_response:
            for b in book.get("bookmakers", book.get("bookmaker", [])) if False else []:
                pass
        # The actual API structure is nested; we'll attempt to extract commonly available markets
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
                for bet in bm.get("bets", bm.get("values", [])):
                    label = bet.get("name") or bet.get("label") or ""
                    if "1X2" in label.upper() or "MATCH ODDS" in label.upper():
                        for v in bet.get("values", bet.get("values", [])):
                            key = v.get("value", "") or v.get("odd", "")
                            odd = v.get("odd") or v.get("value")
                            # skip if impossible shape
                    if "BTTS" in label.upper() or "BOTH" in label.upper():
                        for v in bet.get("values", []):
                            key = v.get("value", "").lower()
                            odd = v.get("odd")
                            if not odd:
                                continue
                            try:
                                odd_f = float(odd)
                                if "yes" in key or "y" == key:
                                    markets.setdefault("btts_yes_odds", odd_f)
                                elif "no" in key:
                                    markets.setdefault("btts_no_odds", odd_f)
                            except Exception:
                                continue
                    if "OVER/UNDER" in label.upper() or "TOTAL" in label.upper():
                        for v in bet.get("values", []):
                            key = v.get("value", "").lower()
                            odd = v.get("odd")
                            if "2.5" in key and odd:
                                try:
                                    odd_f = float(odd)
                                    if "over" in key:
                                        markets.setdefault("over25_odds", odd_f)
                                    elif "under" in key:
                                        markets.setdefault("under25_odds", odd_f)
                                except Exception:
                                    continue
        # Convert odds to implied probabilities
        market_probs = {}
        if markets.get("btts_yes_odds"):
            market_probs["btts_yes"] = round(1.0 / markets["btts_yes_odds"], 4)
            market_probs["btts_no"] = round(1.0 - market_probs["btts_yes"], 4)
        if markets.get("over25_odds"):
            market_probs["over25"] = round(1.0 / markets["over25_odds"], 4)
            market_probs["under25"] = round(1.0 - market_probs["over25"], 4)
        # Note: 1X2 odds parsing is complex; we leave it for extended implementation.
        return market_probs

    def compute_edges_and_kelly(self, model_probs: Dict[str, float], market_probs: Dict[str, float], market_odds: Dict[str, float] = None) -> Dict:
        """
        For each market available in market_probs, compute edge and Kelly stake %.
        edge = model_prob * odds - 1   (per provided spec)
        kelly = edge / (odds - 1) if odds>1 and edge>0 else 0
        """
        res = {}
        # Expect market_probs contain implied probs; for Kelly we need odds. Try infer odds from probs if not provided.
        for m in ["btts_yes", "over25"]:
            if m in model_probs and m in market_probs:
                implied = market_probs[m]
                if implied <= 0:
                    continue
                odds = (1.0 / implied) if implied > 0 else None
                model_p = model_probs[m]
                if odds and odds > 1:
                    edge = model_p * odds - 1.0
                    kelly = (edge / (odds - 1.0)) if (odds - 1.0) > 0 and edge > 0 else 0.0
                    stake = round(max(0.0, min(1.0, kelly)) * BANKROLL_DAILY, 2)
                    res[m] = {
                        "market_prob": round(implied, 4),
                        "model_prob": round(model_p, 4),
                        "odds": round(odds, 3),
                        "edge": round(edge, 4),
                        "kelly_frac": round(kelly, 4),
                        "stake_recom": stake,
                    }
        return res
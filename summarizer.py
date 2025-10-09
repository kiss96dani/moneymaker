"""
Summarizer: build top-N lists per market from analysis results.

For each fixture:
 - 1X2: pick the strongest 1X2 outcome (home/draw/away) and its probability,
   then select top N fixtures by that probability.
 - BTTS: pick the stronger side (yes/no) and rank by probability.
 - Over/Under (2.5): pick the stronger side (over/under) and rank by probability.

Only tips with odds >= MIN_ODDS are included in the output.

Returns a dict with three keys: '1x2', 'btts', 'over25', each a list of top entries.
Each entry contains fixture_id, teams, league, kickoff, chosen_outcome, probability, and source fields.
"""
from datetime import datetime, timezone
from typing import List, Dict, Any
from pathlib import Path
import os

# Minimum odds threshold for tips (default 1.85)
MIN_ODDS = float(os.getenv("MIN_ODDS", "1.85"))

def _fixture_brief(r: Dict[str, Any]):
    brief = {
        "fixture_id": r.get("fixture_id"),
        "kickoff_utc": r.get("kickoff_utc"),
        "kickoff_local": r.get("kickoff_local"),
        "league_id": r.get("league_id"),
        "league_name": r.get("league_name"),
        "home_team": r.get("home_team"),
        "away_team": r.get("away_team"),
    }
    return brief

def summarize_results(results: List[Dict[str, Any]], top_n: int = 3, require_tippmix: bool = False) -> Dict[str, Any]:
    # collectors
    one_x_two_candidates = []
    btts_candidates = []
    over_candidates = []
    
    # Initialize Tippmix lookup if required
    tippmix_lookup = None
    if require_tippmix:
        try:
            from tippmix_lookup import get_lookup
            tippmix_lookup = get_lookup()
            from utils import log
            log("INFO", "Tippmix filtering enabled")
        except Exception as e:
            from utils import log
            log("WARNING", f"Could not initialize Tippmix lookup: {e}")

    for r in results:
        mp = r.get("model_probs", {})
        brief = _fixture_brief(r)
        edge_kelly = r.get("edge_kelly", {})
        market = r.get("market", {})
        market_odds = market.get("market_odds", {})
        
        # Check Tippmix availability if required
        if require_tippmix and tippmix_lookup:
            home_team = r.get("home_team", "")
            away_team = r.get("away_team", "")
            if not tippmix_lookup.is_on_tippmix(home_team, away_team):
                continue  # Skip this fixture

        # 1X2: choose strongest side
        home_p = mp.get("home")
        draw_p = mp.get("draw")
        away_p = mp.get("away")
        if home_p is not None and draw_p is not None and away_p is not None:
            best_side = "home"
            best_prob = home_p
            if draw_p > best_prob:
                best_side = "draw"
                best_prob = draw_p
            if away_p > best_prob:
                best_side = "away"
                best_prob = away_p
            
            # Get edge/Kelly info for the chosen outcome
            ek_info = edge_kelly.get(best_side, {})
            tip_data = {
                **brief,
                "chosen_outcome": best_side,
                "probability": round(float(best_prob), 4),
                "source": "model_probs"
            }
            
            # Add market odds and edge info if available
            if ek_info:
                tip_data["market_odds"] = ek_info.get("odds")
                tip_data["implied_prob"] = ek_info.get("market_prob")
                tip_data["edge"] = ek_info.get("edge")
                tip_data["kelly_frac"] = ek_info.get("kelly_frac")
                tip_data["stake_recom"] = ek_info.get("stake_recom")
            
            # Only add if odds >= MIN_ODDS (or no odds available)
            if tip_data.get("market_odds") is None or tip_data["market_odds"] >= MIN_ODDS:
                one_x_two_candidates.append(tip_data)

        # BTTS: choose stronger side
        btts_yes = mp.get("btts_yes")
        btts_no = mp.get("btts_no")
        if btts_yes is not None and btts_no is not None:
            if btts_yes >= btts_no:
                chosen = "btts_yes"
                prob = btts_yes
            else:
                chosen = "btts_no"
                prob = btts_no
            
            # Get edge/Kelly info for the chosen outcome
            ek_info = edge_kelly.get(chosen, {})
            tip_data = {
                **brief,
                "chosen_outcome": chosen,
                "probability": round(float(prob), 4),
                "source": "model_probs"
            }
            
            # Add market odds and edge info if available
            if ek_info:
                tip_data["market_odds"] = ek_info.get("odds")
                tip_data["implied_prob"] = ek_info.get("market_prob")
                tip_data["edge"] = ek_info.get("edge")
                tip_data["kelly_frac"] = ek_info.get("kelly_frac")
                tip_data["stake_recom"] = ek_info.get("stake_recom")
            
            # Only add if odds >= MIN_ODDS (or no odds available)
            if tip_data.get("market_odds") is None or tip_data["market_odds"] >= MIN_ODDS:
                btts_candidates.append(tip_data)

        # Over/Under 2.5: choose stronger side
        over = mp.get("over25")
        under = mp.get("under25")
        if over is not None and under is not None:
            if over >= under:
                chosen = "over25"
                prob = over
            else:
                chosen = "under25"
                prob = under
            
            # Get edge/Kelly info for the chosen outcome
            ek_info = edge_kelly.get(chosen, {})
            tip_data = {
                **brief,
                "chosen_outcome": chosen,
                "probability": round(float(prob), 4),
                "source": "model_probs"
            }
            
            # Add market odds and edge info if available
            if ek_info:
                tip_data["market_odds"] = ek_info.get("odds")
                tip_data["implied_prob"] = ek_info.get("market_prob")
                tip_data["edge"] = ek_info.get("edge")
                tip_data["kelly_frac"] = ek_info.get("kelly_frac")
                tip_data["stake_recom"] = ek_info.get("stake_recom")
            
            # Only add if odds >= MIN_ODDS (or no odds available)
            if tip_data.get("market_odds") is None or tip_data["market_odds"] >= MIN_ODDS:
                over_candidates.append(tip_data)

    # sort and pick top_n
    one_x_two_sorted = sorted(one_x_two_candidates, key=lambda x: x["probability"], reverse=True)[:top_n]
    btts_sorted = sorted(btts_candidates, key=lambda x: x["probability"], reverse=True)[:top_n]
    over_sorted = sorted(over_candidates, key=lambda x: x["probability"], reverse=True)[:top_n]

    # Build final report with metadata
    now = datetime.now(timezone.utc).isoformat()
    report = {
        "generated_at": now,
        "top_n": top_n,
        "1x2": one_x_two_sorted,
        "btts": btts_sorted,
        "over_under_2_5": over_sorted
    }

    # print human-readable summary to stdout
    print("\n=== Top markets summary ===")
    def _print_section(title: str, items: List[Dict[str, Any]]):
        print(f"\n{title}:")
        if not items:
            print("  (no data)")
            return
        for i, it in enumerate(items, start=1):
            teams = f"{it['home_team']} vs {it['away_team']}"
            prob_str = f"{it['probability']*100:.1f}%"
            
            # Add market odds and edge info if available
            extra_info = []
            if it.get("market_odds"):
                extra_info.append(f"odds: {it['market_odds']:.2f}")
            if it.get("edge") is not None:
                extra_info.append(f"edge: {it['edge']:.2%}")
            if it.get("kelly_frac") is not None:
                extra_info.append(f"kelly: {it['kelly_frac']:.2%}")
            
            extra_str = f" ({', '.join(extra_info)})" if extra_info else ""
            print(f"  {i}. {teams} ({it['league_name']}) - {it['chosen_outcome']} @ {prob_str}{extra_str} [fixture {it['fixture_id']}]")

    _print_section("1X2 (best single outcome per fixture)", one_x_two_sorted)
    _print_section("BTTS (best side per fixture)", btts_sorted)
    _print_section("Over/Under 2.5 (best side per fixture)", over_sorted)
    print("===========================\n")

    return report
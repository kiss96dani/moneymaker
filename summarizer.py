"""
Summarizer: build top-N lists per market from analysis results.

For each fixture:
 - 1X2: pick the strongest 1X2 outcome (home/draw/away) and its probability,
   then select top N fixtures by that probability.
 - BTTS: pick the stronger side (yes/no) and rank by probability.
 - Over/Under (2.5): pick the stronger side (over/under) and rank by probability.

Returns a dict with three keys: '1x2', 'btts', 'over25', each a list of top entries.
Each entry contains fixture_id, teams, league, kickoff, chosen_outcome, probability, and source fields.
"""
from datetime import datetime, timezone
from typing import List, Dict, Any
from pathlib import Path

def _fixture_brief(r: Dict[str, Any]):
    return {
        "fixture_id": r.get("fixture_id"),
        "kickoff_utc": r.get("kickoff_utc"),
        "league_id": r.get("league_id"),
        "league_name": r.get("league_name"),
        "home_team": r.get("home_team"),
        "away_team": r.get("away_team"),
    }

def summarize_results(results: List[Dict[str, Any]], top_n: int = 3) -> Dict[str, Any]:
    # collectors
    one_x_two_candidates = []
    btts_candidates = []
    over_candidates = []

    for r in results:
        mp = r.get("model_probs", {})
        brief = _fixture_brief(r)

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
            one_x_two_candidates.append({
                **brief,
                "chosen_outcome": best_side,
                "probability": round(float(best_prob), 4),
                "source": "model_probs"
            })

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
            btts_candidates.append({
                **brief,
                "chosen_outcome": chosen,
                "probability": round(float(prob), 4),
                "source": "model_probs"
            })

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
            over_candidates.append({
                **brief,
                "chosen_outcome": chosen,
                "probability": round(float(prob), 4),
                "source": "model_probs"
            })

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
            print(f"  {i}. {teams} ({it['league_name']}) - {it['chosen_outcome']} @ {it['probability']*100:.1f}% [fixture {it['fixture_id']}]")

    _print_section("1X2 (best single outcome per fixture)", one_x_two_sorted)
    _print_section("BTTS (best side per fixture)", btts_sorted)
    _print_section("Over/Under 2.5 (best side per fixture)", over_sorted)
    print("===========================\n")

    return report
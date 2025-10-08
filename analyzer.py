from typing import List, Dict, Any
from collections import deque
import math

def _result_from_score(home_sc: int, away_sc: int, is_home: bool):
    if home_sc == away_sc:
        return "D"
    if is_home:
        return "W" if home_sc > away_sc else "L"
    else:
        return "W" if away_sc > home_sc else "L"

class Analyzer:
    def __init__(self):
        pass

    def extract_fixture_meta(self, fixture_raw: Dict[str, Any]) -> Dict[str, Any]:
        fx = fixture_raw.get("fixture", {})
        teams = fixture_raw.get("teams", {})
        league = fixture_raw.get("league", {})
        return {
            "fixture_id": fx.get("id"),
            "kickoff_utc": fx.get("date"),
            "league_id": league.get("id"),
            "league_name": league.get("name"),
            "home_team_id": teams.get("home", {}).get("id"),
            "home_team_name": teams.get("home", {}).get("name"),
            "away_team_id": teams.get("away", {}).get("id"),
            "away_team_name": teams.get("away", {}).get("name"),
        }

    def calculate_last_n_form(self, fixtures: List[Dict[str, Any]], team_id: int, n: int = 5) -> Dict[str, Any]:
        """
        Calculate last n form summary from fixtures returned by API-Football filtered to completed fixtures.
        Returns:
          {
            "form_string": "WWDLW",
            "goals_per_match": 1.8,
            "goals_against_per_match": 1.2,
            "matches_count": 5
          }
        """
        completed = []
        for f in fixtures:
            status = f.get("fixture", {}).get("status", {}).get("short")
            if status not in ("FT", "AET", "PEN"):
                continue
            # determine team's score
            teams = f.get("teams", {})
            home = teams.get("home", {})
            away = teams.get("away", {})
            home_id = home.get("id")
            away_id = away.get("id")
            scores = f.get("goals", {})
            home_g = scores.get("home", 0)
            away_g = scores.get("away", 0)
            completed.append({
                "home_id": home_id,
                "away_id": away_id,
                "home_goals": home_g,
                "away_goals": away_g,
            })

        # order: API usually returns newest first; make newest last to read chronological
        completed = list(reversed(completed))
        last_n = completed[-n:] if len(completed) >= n else completed

        form = []
        goals_for = 0
        goals_against = 0
        for m in last_n:
            is_home = (m["home_id"] == team_id)
            gf = m["home_goals"] if is_home else m["away_goals"]
            ga = m["away_goals"] if is_home else m["home_goals"]
            res = _result_from_score(m["home_goals"], m["away_goals"], is_home)
            form.append(res)
            goals_for += gf
            goals_against += ga

        matches = len(last_n) or 0
        gpm = round(goals_for / matches, 2) if matches else 0.0
        gapm = round(goals_against / matches, 2) if matches else 0.0
        form_string = "".join(form[-5:])  # last up to 5
        return {
            "form_string": form_string or "",
            "goals_per_match": gpm,
            "goals_against_per_match": gapm,
            "matches_count": matches,
        }
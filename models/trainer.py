"""
Model training module for ML-based predictions.

Fetches historical fixtures, engineers features, and trains LogisticRegression models
for 1X2, BTTS, and Over/Under 2.5 markets.
"""
import os
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

from utils import log

HOME_ADV = float(os.getenv("HOME_ADV", "0.20"))


class ModelTrainer:
    def __init__(self, client):
        """
        Initialize trainer with an APIFootballClient instance.
        
        Args:
            client: APIFootballClient instance for fetching historical data
        """
        self.client = client
        self.models_dir = Path("models")
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def _calculate_form_score(self, form_string: str) -> float:
        """
        Calculate weighted form score from result string.
        W=1, D=0.5, L=0. Recent matches weighted more.
        
        Args:
            form_string: String like "WWDLW" representing recent results
            
        Returns:
            Weighted form score
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

    async def fetch_historical_fixtures(
        self, 
        league_ids: List[int], 
        from_date: str, 
        to_date: str
    ) -> List[Dict[str, Any]]:
        """
        Fetch historical fixtures for specified leagues and date range.
        
        Args:
            league_ids: List of league IDs to fetch
            from_date: Start date in YYYY-MM-DD format
            to_date: End date in YYYY-MM-DD format
            
        Returns:
            List of fixture data dictionaries
        """
        all_fixtures = []
        
        for league_id in league_ids:
            try:
                log("INFO", f"Fetching fixtures for league {league_id} from {from_date} to {to_date}")
                fixtures = await self.client.get_fixtures_by_league_and_date(
                    league_id, from_date, to_date
                )
                
                # Filter to only completed matches
                completed = [
                    f for f in fixtures
                    if f.get("fixture", {}).get("status", {}).get("short") in ("FT", "AET", "PEN")
                ]
                
                all_fixtures.extend(completed)
                log("INFO", f"League {league_id}: fetched {len(completed)} completed fixtures")
                await asyncio.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                log("ERROR", f"Failed to fetch fixtures for league {league_id}: {e}")
                continue
        
        log("INFO", f"Total historical fixtures fetched: {len(all_fixtures)}")
        return all_fixtures

    def _extract_result_labels(self, fixture: Dict[str, Any]) -> Tuple[int, int, int]:
        """
        Extract 1X2, BTTS, Over2.5 labels from fixture result.
        
        Returns:
            Tuple of (result_1x2, btts, over25)
            - result_1x2: 0=home, 1=draw, 2=away
            - btts: 1=yes, 0=no
            - over25: 1=yes, 0=no
        """
        goals = fixture.get("goals", {})
        home_goals = goals.get("home", 0) or 0
        away_goals = goals.get("away", 0) or 0
        
        # 1X2 label
        if home_goals > away_goals:
            result_1x2 = 0  # home win
        elif home_goals == away_goals:
            result_1x2 = 1  # draw
        else:
            result_1x2 = 2  # away win
        
        # BTTS label
        btts = 1 if (home_goals > 0 and away_goals > 0) else 0
        
        # Over 2.5 label
        over25 = 1 if (home_goals + away_goals) > 2.5 else 0
        
        return result_1x2, btts, over25

    async def _get_team_form_at_date(
        self, 
        team_id: int, 
        before_date: datetime,
        n_matches: int = 5
    ) -> Dict[str, float]:
        """
        Get team form statistics before a specific date.
        
        Args:
            team_id: Team ID
            before_date: Date before which to look for matches
            n_matches: Number of recent matches to consider
            
        Returns:
            Dict with goals_per_match, goals_against_per_match, form_score
        """
        try:
            # Fetch recent fixtures for the team
            fixtures = await self.client.get_team_recent_fixtures(team_id, limit=20)
            
            # Filter to matches before the target date and completed
            valid_fixtures = []
            for f in fixtures:
                fixture_date_str = f.get("fixture", {}).get("date", "")
                if not fixture_date_str:
                    continue
                    
                fixture_date = datetime.fromisoformat(fixture_date_str.replace("Z", "+00:00"))
                status = f.get("fixture", {}).get("status", {}).get("short")
                
                if fixture_date < before_date and status in ("FT", "AET", "PEN"):
                    valid_fixtures.append(f)
            
            # Take last n matches
            valid_fixtures = sorted(
                valid_fixtures, 
                key=lambda x: x.get("fixture", {}).get("date", "")
            )[-n_matches:]
            
            if not valid_fixtures:
                return {
                    "goals_per_match": 1.0,
                    "goals_against_per_match": 1.0,
                    "form_score": 0.5
                }
            
            goals_for = 0
            goals_against = 0
            form_results = []
            
            for f in valid_fixtures:
                teams = f.get("teams", {})
                goals = f.get("goals", {})
                
                home_id = teams.get("home", {}).get("id")
                home_goals = goals.get("home", 0) or 0
                away_goals = goals.get("away", 0) or 0
                
                is_home = (home_id == team_id)
                gf = home_goals if is_home else away_goals
                ga = away_goals if is_home else home_goals
                
                goals_for += gf
                goals_against += ga
                
                # Determine result
                if gf > ga:
                    form_results.append('W')
                elif gf == ga:
                    form_results.append('D')
                else:
                    form_results.append('L')
            
            n = len(valid_fixtures)
            form_string = "".join(form_results[-5:])
            
            return {
                "goals_per_match": goals_for / n if n > 0 else 1.0,
                "goals_against_per_match": goals_against / n if n > 0 else 1.0,
                "form_score": self._calculate_form_score(form_string)
            }
            
        except Exception as e:
            log("WARNING", f"Error getting team form for {team_id}: {e}")
            return {
                "goals_per_match": 1.0,
                "goals_against_per_match": 1.0,
                "form_score": 0.5
            }

    async def engineer_features(
        self, 
        fixtures: List[Dict[str, Any]]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Engineer features and labels from historical fixtures.
        
        Args:
            fixtures: List of completed fixture data
            
        Returns:
            Tuple of (X, y_1x2, y_btts, y_over25) DataFrames
        """
        rows = []
        
        for fixture in fixtures:
            try:
                # Extract basic info
                teams = fixture.get("teams", {})
                home_id = teams.get("home", {}).get("id")
                away_id = teams.get("away", {}).get("id")
                
                if not home_id or not away_id:
                    continue
                
                fixture_date_str = fixture.get("fixture", {}).get("date", "")
                if not fixture_date_str:
                    continue
                
                fixture_date = datetime.fromisoformat(fixture_date_str.replace("Z", "+00:00"))
                
                # Get form before this match
                home_form = await self._get_team_form_at_date(home_id, fixture_date)
                await asyncio.sleep(0.05)  # Small delay to avoid rate limits
                
                away_form = await self._get_team_form_at_date(away_id, fixture_date)
                await asyncio.sleep(0.05)
                
                # Extract labels
                result_1x2, btts, over25 = self._extract_result_labels(fixture)
                
                # Build feature row
                row = {
                    'home_goals_per_match': home_form['goals_per_match'],
                    'home_goals_against_per_match': home_form['goals_against_per_match'],
                    'away_goals_per_match': away_form['goals_per_match'],
                    'away_goals_against_per_match': away_form['goals_against_per_match'],
                    'form_score_home': home_form['form_score'],
                    'form_score_away': away_form['form_score'],
                    'home_adv': HOME_ADV,
                    'result_1x2': result_1x2,
                    'btts': btts,
                    'over25': over25
                }
                
                rows.append(row)
                
            except Exception as e:
                log("WARNING", f"Error engineering features for fixture: {e}")
                continue
        
        if not rows:
            raise ValueError("No valid features could be engineered from fixtures")
        
        df = pd.DataFrame(rows)
        
        feature_cols = [
            'home_goals_per_match', 'home_goals_against_per_match',
            'away_goals_per_match', 'away_goals_against_per_match',
            'form_score_home', 'form_score_away', 'home_adv'
        ]
        
        X = df[feature_cols]
        y_1x2 = df['result_1x2']
        y_btts = df['btts']
        y_over25 = df['over25']
        
        log("INFO", f"Engineered {len(X)} feature samples")
        return X, y_1x2, y_btts, y_over25

    def train_models(
        self, 
        X: pd.DataFrame, 
        y_1x2: pd.DataFrame, 
        y_btts: pd.DataFrame, 
        y_over25: pd.DataFrame
    ):
        """
        Train and save LogisticRegression models for all three markets.
        
        Args:
            X: Feature DataFrame
            y_1x2: 1X2 labels (0=home, 1=draw, 2=away)
            y_btts: BTTS labels (0=no, 1=yes)
            y_over25: Over 2.5 labels (0=under, 1=over)
        """
        log("INFO", "Training models...")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train 1X2 model (multinomial)
        log("INFO", "Training 1X2 model (multinomial logistic regression)...")
        model_1x2 = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            max_iter=1000,
            random_state=42
        )
        model_1x2.fit(X_scaled, y_1x2)
        
        # Train BTTS model (binary)
        log("INFO", "Training BTTS model (binary logistic regression)...")
        model_btts = LogisticRegression(
            solver='lbfgs',
            max_iter=1000,
            random_state=42
        )
        model_btts.fit(X_scaled, y_btts)
        
        # Train Over/Under 2.5 model (binary)
        log("INFO", "Training Over/Under 2.5 model (binary logistic regression)...")
        model_over25 = LogisticRegression(
            solver='lbfgs',
            max_iter=1000,
            random_state=42
        )
        model_over25.fit(X_scaled, y_over25)
        
        # Save models
        log("INFO", "Saving models to disk...")
        joblib.dump(scaler, self.models_dir / "feature_scaler.pkl")
        joblib.dump(model_1x2, self.models_dir / "1x2.pkl")
        joblib.dump(model_btts, self.models_dir / "btts.pkl")
        joblib.dump(model_over25, self.models_dir / "over25.pkl")
        
        log("INFO", f"Models saved to {self.models_dir}/")
        
        # Print basic evaluation
        log("INFO", f"1X2 training accuracy: {model_1x2.score(X_scaled, y_1x2):.3f}")
        log("INFO", f"BTTS training accuracy: {model_btts.score(X_scaled, y_btts):.3f}")
        log("INFO", f"Over/Under 2.5 training accuracy: {model_over25.score(X_scaled, y_over25):.3f}")

    async def run_training_pipeline(
        self,
        league_ids: List[int],
        from_date: str,
        to_date: str
    ):
        """
        Complete training pipeline: fetch data, engineer features, train models.
        
        Args:
            league_ids: List of league IDs to train on
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
        """
        log("INFO", f"Starting training pipeline for leagues {league_ids}")
        log("INFO", f"Date range: {from_date} to {to_date}")
        
        # Fetch historical fixtures
        fixtures = await self.fetch_historical_fixtures(league_ids, from_date, to_date)
        
        if len(fixtures) < 50:
            log("WARNING", f"Only {len(fixtures)} fixtures found. Model quality may be poor.")
        
        # Engineer features
        X, y_1x2, y_btts, y_over25 = await self.engineer_features(fixtures)
        
        # Train models
        self.train_models(X, y_1x2, y_btts, y_over25)
        
        log("INFO", "Training pipeline completed successfully!")

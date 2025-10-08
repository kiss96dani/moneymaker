"""
Model wrapper for loading and using trained ML models.

Provides a simple interface to load models and get predictions.
"""
from pathlib import Path
from typing import Dict, Optional, Tuple
import joblib
import numpy as np

from utils import log


class ModelWrapper:
    """Wrapper for trained ML models."""
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize model wrapper.
        
        Args:
            models_dir: Directory containing saved model files
        """
        self.models_dir = Path(models_dir)
        self.scaler = None
        self.model_1x2 = None
        self.model_btts = None
        self.model_over25 = None
        self.loaded = False
        
    def load_models(self) -> bool:
        """
        Load all trained models from disk.
        
        Returns:
            True if all models loaded successfully, False otherwise
        """
        try:
            scaler_path = self.models_dir / "feature_scaler.pkl"
            model_1x2_path = self.models_dir / "1x2.pkl"
            model_btts_path = self.models_dir / "btts.pkl"
            model_over25_path = self.models_dir / "over25.pkl"
            
            # Check all files exist
            if not all([
                scaler_path.exists(),
                model_1x2_path.exists(),
                model_btts_path.exists(),
                model_over25_path.exists()
            ]):
                log("WARNING", "Not all model files found. ML mode unavailable.")
                return False
            
            # Load models
            self.scaler = joblib.load(scaler_path)
            self.model_1x2 = joblib.load(model_1x2_path)
            self.model_btts = joblib.load(model_btts_path)
            self.model_over25 = joblib.load(model_over25_path)
            
            self.loaded = True
            log("INFO", "ML models loaded successfully")
            return True
            
        except Exception as e:
            log("ERROR", f"Failed to load models: {e}")
            self.loaded = False
            return False
    
    def predict_proba(self, features: Dict[str, float]) -> Optional[Dict[str, float]]:
        """
        Predict probabilities for all three markets.
        
        Args:
            features: Dict with keys:
                - home_goals_per_match
                - home_goals_against_per_match
                - away_goals_per_match
                - away_goals_against_per_match
                - form_score_home
                - form_score_away
                - home_adv
        
        Returns:
            Dict with probabilities for all outcomes, or None if models not loaded
        """
        if not self.loaded:
            log("WARNING", "Models not loaded. Call load_models() first.")
            return None
        
        try:
            # Prepare feature vector
            feature_order = [
                'home_goals_per_match',
                'home_goals_against_per_match',
                'away_goals_per_match',
                'away_goals_against_per_match',
                'form_score_home',
                'form_score_away',
                'home_adv'
            ]
            
            X = np.array([[features[k] for k in feature_order]])
            X_scaled = self.scaler.transform(X)
            
            # Get predictions
            probs_1x2 = self.model_1x2.predict_proba(X_scaled)[0]  # [home, draw, away]
            probs_btts = self.model_btts.predict_proba(X_scaled)[0]  # [no, yes]
            probs_over25 = self.model_over25.predict_proba(X_scaled)[0]  # [under, over]
            
            # Format results
            result = {
                "home": round(float(probs_1x2[0]), 4),
                "draw": round(float(probs_1x2[1]), 4),
                "away": round(float(probs_1x2[2]), 4),
                "btts_no": round(float(probs_btts[0]), 4),
                "btts_yes": round(float(probs_btts[1]), 4),
                "under25": round(float(probs_over25[0]), 4),
                "over25": round(float(probs_over25[1]), 4),
            }
            
            return result
            
        except Exception as e:
            log("ERROR", f"Prediction failed: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if models are loaded and available."""
        return self.loaded
